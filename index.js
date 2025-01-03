var express = require('express');
var app = express();

app.get("/", (req, res) => {
    res.send("<h1>Welcome</h1>");
});


app.listen(3004, () => {
    console.log("Application listening on port 3004");
});