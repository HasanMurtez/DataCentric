const express = require('express');
const router = express.Router();

let students = [
    { sid: 'G001', name: 'Sean Smith', age: 32 },
    { sid: 'G002', name: 'Alison Conners', age: 23 },
    { sid: 'G003', name: 'Thomas Murphy', age: 19 },
    { sid: 'G004', name: 'Anne Greene', age: 23 },
    { sid: 'G005', name: 'Tom Riddle', age: 27 },
    { sid: 'G006', name: 'Brian Collins', age: 38 },
    { sid: 'G007', name: 'Fiona O Hehir', age: 30 },
    { sid: 'G008', name: 'George Johnson', age: 24 },
    { sid: 'G009', name: 'Albert Newton', age: 31 },
    { sid: 'G0010', name: 'Marie Yeats', age: 21 },
    { sid: 'G0011', name: 'Johnathon Small', age: 22 },
    { sid: 'G0012', name: 'Barbara Harris', age: 23 },
    { sid: 'G0013', name: 'Oliver Flanagan', age: 19 },
    { sid: 'G0014', name: 'Neil Blaney', age: 34 },
    { sid: 'G0015', name: 'Nigel Delaney', age: 19 },
    { sid: 'G0016', name: 'Johhny Connors', age: 29 },
    { sid: 'G0017', name: 'Bill Turpin', age: 18 },
    { sid: 'G0018', name: 'Amanda Knox', age: 23 },
    { sid: 'G0019', name: 'James Joyce', age: 39 },
    { sid: 'G0020', name: 'Alice L Estrange', age: 32 },
];


const findStudent = (sid) => students.find((student) => student.sid === sid);
const authenticateStudent = ({ sid, name, age }, isNew = true) => {
    const errors = [];
    if (isNew) {
        if (!sid || sid.length !== 4) errors.push("Student ID must be exactly 4 characters.");
        if (students.some((s) => s.sid === sid)) errors.push("Student ID already exists.");
    }
    if (!name || name.length < 2) errors.push("Name should be at least 2 characters.");
    if (!age || age < 18) errors.push("Age should be 18 or older.");
    return errors;
};

//display all students
router.get('/', (req, res) => {
    res.render('students', { students, title: "Students" });
});

router.get('/edit/:sid', (req, res) => {
    const student = findStudent(req.params.sid);
    if (!student) return res.status(404).send(`<h1>Error: Student not found</h1>`);
    res.render('editStudents', { student, errors: [] });
});

router.post('/edit/:sid', (req, res) => {
    const student = findStudent(req.params.sid);
    if (!student) return res.status(404).send(`<h1>Error: Student not found</h1>`);

    const { name, age } = req.body;
    const errors = authenticateStudent({ name, age }, false);

    if (errors.length) {
        return res.render('editStudents', { student: { ...student, name, age }, errors });
    }

    student.name = name;
    student.age = age;
    res.redirect('/students');
});

router.get('/add', (req, res) => {
    res.render('addStudent', { errors: [], student: {} });
});

//adding a new student
router.post('/add', (req, res) => {
    const { sid, name, age } = req.body;
    const errors = authenticateStudent({ sid, name, age });

    if (errors.length) {
        return res.render('addStudent', { errors, student: { sid, name, age } });
    }

    students.push({ sid, name, age });
    res.redirect('/students');
});

module.exports = router;
