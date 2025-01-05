const MongoClient = require('mongodb').MongoClient;
let db;
let coll;

//connect to MongoDB
MongoClient.connect('mongodb://127.0.0.1:27017')
    .then((client) => {
        db = client.db('proj2024MongoDB'); //database name
        coll = db.collection('lecturers'); //collection name
    })
    .catch((error) => {
        console.log('MongoDB connection error:', error.message);
    });

//function to retrieve all lecturers sorted by lecturerId
const findLecturers = () => {
    return new Promise((resolve, reject) => {
        coll.find().sort({ _id: 1 }).toArray()
            .then((documents) => {
                resolve(documents);
            })
            .catch((error) => {
                reject(error);
            });
    });
};

module.exports = { findLecturers };
