var express = require('express');
var router = express.Router();

var grades = [
    { student: 'Albert Newton', module: 'Algebra', grade: 49 },
    { student: 'Albert Newton', module: 'Mechanics of Fluids', grade: 78 },
    { student: 'Alice L Estrange', module: null, grade: null },
    { student: 'Alison Conners', module: 'Mechanics of Fluids', grade: 72 },
    { student: 'Alison Conners', module: 'Mechanics of Solids', grade: 79 },
    { student: 'Amanda Knox', module: 'Long Division', grade: 32 },
    { student: 'Amanda Knox', module: 'Times Tables', grade: 65 },
    { student: 'Amanda Knox', module: 'Algebra', grade: 77 },
    { student: 'Anne Greene', module: 'Poetry', grade: 45 },
    { student: 'Anne Greene', module: 'Creative Writing', grade: 56 },
    { student: 'Anne Greene', module: 'Shakespeare', grade: 71 },
    { student: 'Barbara Harris', module: null, grade: null },
    { student: 'Bill Turpin', module: 'Shakespeare', grade: 68 },
    { student: 'Brian Collins', module: 'Algebra', grade: 28 },
    { student: 'Brian Collins', module: 'Times Tables', grade: 91 },
    { student: 'Brian Collins', module: 'Long Division', grade: 92 },
    { student: 'Fiona O Hehir', module: 'Creative Writing', grade: 55 },
    { student: 'George Johnson', module: 'Poetry', grade: 72 },
    { student: 'George Johnson', module: 'Creative Writing', grade: 82 },
    { student: 'James Joyce', module: 'Mobile Applications Development', grade: 32 },
    { student: 'Johnny Connors', module: 'Mechanics of Fluids', grade: 35 },
    { student: 'Johnny Connors', module: 'Mechanics of Solids', grade: 52 },
];

//display the grades
router.get('/', (req, res) => {
    res.render('grades', { grades, title: 'Grades' });
});

module.exports = router;