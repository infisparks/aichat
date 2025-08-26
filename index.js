// index.js - A Node.js and Express backend for the Infispark AI Chatbot 
// using TensorFlow.js and Firebase

require('dotenv').config();
const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const natural = require('natural');
const admin = require('firebase-admin');
const fs = require('fs');
const crypto = require('crypto');

const app = express();
const port = 3000;

// --- Configuration ---
const API_PASSWORD = "mudassirs472"; // Your chosen password
const MODEL_PATH = 'file://./model';
const METADATA_PATH = './model/metadata.json';

// --- Firebase Initialization (from ENV variables) ---
admin.initializeApp({
  credential: admin.credential.cert({
    type: process.env.TYPE,
    project_id: process.env.PROJECT_ID,
    private_key_id: process.env.PRIVATE_KEY_ID,
    private_key: process.env.PRIVATE_KEY.replace(/\\n/g, '\n'),
    client_email: process.env.CLIENT_EMAIL,
    client_id: process.env.CLIENT_ID,
    auth_uri: process.env.AUTH_URI,
    token_uri: process.env.TOKEN_URI,
    auth_provider_x509_cert_url: process.env.AUTH_PROVIDER_X509_CERT_URL,
    client_x509_cert_url: process.env.CLIENT_X509_CERT_URL,
    universe_domain: process.env.UNIVERSE_DOMAIN
  }),
  databaseURL: `https://${process.env.PROJECT_ID}-default-rtdb.firebaseio.com`
});
const db = admin.database();

// Middleware to parse JSON bodies
app.use(express.json());

// === Global variables ===
let model;
let words = [];
let classes = [];
let intentsData;
let currentDataHash = null;
const tokenizer = new natural.WordTokenizer();

// === Authentication Middleware ===
const checkPassword = (req, res, next) => {
    const providedPassword = req.body.password || req.headers['authorization'];
    if (!providedPassword) {
        return res.status(401).json({ error: 'Unauthorized: No password provided.' });
    }
    const passwordToCheck = providedPassword.startsWith('Bearer ') 
        ? providedPassword.substring(7) 
        : providedPassword;

    if (passwordToCheck !== API_PASSWORD) {
        return res.status(403).json({ error: 'Forbidden: Incorrect password.' });
    }
    next();
};

// --- Helper Functions ---
function bagOfWords(input, vocabulary) {
    const bag = Array(vocabulary.length).fill(0);
    const tokenizedInput = tokenizer.tokenize(input.toLowerCase());
    tokenizedInput.forEach(word => {
        const index = vocabulary.indexOf(word);
        if (index > -1) {
            bag[index] = 1;
        }
    });
    return tf.tensor1d(bag);
}

function createDataHash(data) {
    return crypto.createHash('md5').update(JSON.stringify(data)).digest('hex');
}

// === Core Model Functions ===
async function trainAndSaveModel(intents) {
    console.log('Starting data preprocessing...');
    words = [];
    classes = [];
    const documents = [];
    intentsData = intents;

    intents.intents.forEach(intent => {
        const tag = intent.tag;
        if (!classes.includes(tag)) {
            classes.push(tag);
        }
        intent.patterns.forEach(pattern => {
            const tokenizedWords = tokenizer.tokenize(pattern.toLowerCase());
            words.push(...tokenizedWords);
            documents.push({ words: tokenizedWords, tag: tag });
        });
    });

    words = [...new Set(words.filter(word => word.length > 1))].sort();
    classes = [...new Set(classes)].sort();

    console.log(`Data preprocessed: ${words.length} words, ${classes.length} classes.`);

    const trainingData = [], outputData = [];
    documents.forEach(doc => {
        trainingData.push(bagOfWords(doc.words.join(" "), words).arraySync());
        const outputRow = Array(classes.length).fill(0);
        outputRow[classes.indexOf(doc.tag)] = 1;
        outputData.push(outputRow);
    });

    const xs = tf.tensor2d(trainingData);
    const ys = tf.tensor2d(outputData);

    const newModel = tf.sequential();
    newModel.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [xs.shape[1]] }));
    newModel.add(tf.layers.dropout({ rate: 0.5 }));
    newModel.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    newModel.add(tf.layers.dropout({ rate: 0.5 }));
    newModel.add(tf.layers.dense({ units: classes.length, activation: 'softmax' }));

    newModel.compile({ optimizer: tf.train.adam(), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    console.log('Starting model training...');
    await newModel.fit(xs, ys, {
        epochs: 200,
        batchSize: 5,
        shuffle: true,
        callbacks: { onEpochEnd: (epoch, logs) => console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, acc = ${logs.acc.toFixed(4)}`) }
    });
    console.log('Training complete!');
    model = newModel;

    console.log('Saving model and metadata...');
    await model.save(MODEL_PATH);
    if (!fs.existsSync('./model')) fs.mkdirSync('./model');
    fs.writeFileSync(METADATA_PATH, JSON.stringify({ words, classes }));
    console.log('Model and metadata saved.');
}

async function loadModelAndMetadata() {
    try {
        if (fs.existsSync('./model/model.json') && fs.existsSync(METADATA_PATH)) {
            console.log('Found saved model. Loading...');
            model = await tf.loadLayersModel(`${MODEL_PATH}/model.json`);
            const metadata = JSON.parse(fs.readFileSync(METADATA_PATH, 'utf-8'));
            words = metadata.words;
            classes = metadata.classes;
            console.log('Model and metadata loaded.');
            return true;
        }
    } catch (error) {
        console.error('Could not load saved model:', error);
    }
    console.log('No saved model found.');
    return false;
}

// === Main App Start Function ===
async function startApp() {
    await loadModelAndMetadata();

    app.listen(port, () => {
        console.log(`Server listening at http://localhost:${port}`);
        console.log(model ? 'Chatbot is ready with a pre-loaded model.' : 'Waiting for Firebase data to train model...');
    });

    const intentsRef = db.ref('intents');
    intentsRef.on('value', async (snapshot) => {
        console.log('\nFirebase data change detected.');
        const newIntents = snapshot.val();

        if (!newIntents || !newIntents.intents) {
            console.error('Invalid data from Firebase. Waiting for valid data...');
            return;
        }

        const newHash = createDataHash(newIntents);
        if (newHash !== currentDataHash) {
            console.log('Data has changed. Retraining model...');
            currentDataHash = newHash;
            await trainAndSaveModel(newIntents);
            console.log('Model is now updated and ready.');
        } else {
            if (!intentsData) intentsData = newIntents;
            console.log('Data is unchanged. No retraining needed.');
        }
    }, (error) => {
        console.error('Firebase listener error:', error);
        process.exit(1);
    });
}

// === API Endpoints ===

// SECURED: Chatbot API Endpoint
app.post('/api/chat', checkPassword, async (req, res) => {
    try {
        if (!model || !intentsData) {
            return res.status(503).json({ error: 'Model is not ready.' });
        }
        if (!req.body.message) {
            return res.status(400).json({ error: 'Invalid request: "message" key is required.' });
        }

        const userMessage = req.body.message;
        const inputBag = bagOfWords(userMessage, words);
        const prediction = model.predict(tf.expandDims(inputBag, 0));
        const predictionData = await prediction.data();
        const maxProbability = Math.max(...predictionData);
        const predictedIndex = predictionData.indexOf(maxProbability);
        let tag = classes[predictedIndex];

        if (maxProbability < 0.7) tag = "default";

        const matchingIntent = intentsData.intents.find(intent => intent.tag === tag);
        const response = matchingIntent.responses[Math.floor(Math.random() * matchingIntent.responses.length)];

        res.json({ response, confidence: maxProbability.toFixed(2) });
    } catch (error) {
        console.error('Error in /api/chat:', error);
        res.status(500).json({ error: 'Internal server error.' });
    }
});

// SECURED: Endpoint to upload and MERGE intents data
app.post('/api/update-intents', checkPassword, async (req, res) => {
    try {
        const newIntentsData = req.body.intentsData;

        if (!newIntentsData || !newIntentsData.intents || !Array.isArray(newIntentsData.intents)) {
            console.error("Validation Error: Invalid JSON structure received.");
            return res.status(400).json({ 
                error: 'Invalid JSON structure. The body must contain an "intentsData" key with an "intents" array.' 
            });
        }

        const intentsRef = db.ref('intents');
        const snapshot = await intentsRef.once('value');
        let existingData = snapshot.val();

        if (!existingData || !existingData.intents) {
            existingData = { intents: [] };
        }

        const intentsMap = new Map(existingData.intents.map(intent => [intent.tag, intent]));
        newIntentsData.intents.forEach(newIntent => {
            intentsMap.set(newIntent.tag, newIntent);
        });

        const mergedIntentsArray = Array.from(intentsMap.values());
        await intentsRef.set({ intents: mergedIntentsArray });

        res.status(200).json({ 
            message: 'Intents data updated/merged successfully. The model will now retrain automatically.' 
        });

    } catch (error) {
        console.error('Error in /api/update-intents:', error);
        res.status(500).json({ error: 'Failed to update intents in Firebase.' });
    }
});

// Root endpoint
app.get('/', (req, res) => {
    res.send('Infispark TensorFlow Chatbot API is running!');
});

// --- Start the application ---
startApp();
