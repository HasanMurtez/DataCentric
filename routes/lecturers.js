const express = require('express');
const router = express.Router();
const dao = require('../dao');

//display lecturers
router.get('/', (req, res) => {
    dao.findLecturers()
        .then((lecturers) => {
            res.render('lecturers', { lecturers, title: 'Lecturers' });
        })
        .catch((error) => {
            console.log('error retrieving lecturers:', error.message);
            res.status(500).send('internal Server Error');
        });
});

module.exports = router;
