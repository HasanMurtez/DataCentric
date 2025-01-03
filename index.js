var express = require('express');
var app = express();

app.set('view engine', 'ejs');
app.set('views', './views');
app.use('/students', require('./routes/students'));

app.get("/", (req, res) => {
    res.render("index");
});

app.listen(3004, () => {
    console.log("Running on port 3004");
});
