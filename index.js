var express = require('express');
var app = express();

app.set('view engine', 'ejs');
app.set('views', './views');

app.use(express.urlencoded({ extended: true }));

// Routes
app.use('/students', require('./routes/students'));
app.use('/grades', require('./routes/grades'));

// Home Page
app.get('/', (req, res) => {
    res.render('index');
});

// Start the server
app.listen(3004, () => {
    console.log('Running on port 3004');
});
