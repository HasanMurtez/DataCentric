var express = require('express');
var router = express.Router();

var students = [
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

//display all students
router.get('/', (req, res) => {
    res.render('students', { students: students, title: "Students" });
});

router.get('/edit/:sid', (req, res) => {
    const studentId = req.params.sid;
    const student = students.find(s => s.sid === studentId); // Find the student by ID

    if (!student) {
        return res.send(`<h1>Error: Student with ID ${studentId} not found</h1>`);
    }

    res.render('editStudents', { student, errors: [] }); 
});

router.post('/edit/:sid', (req, res) => {
    const studentId = req.params.sid;
    const { name, age } = req.body; // Get updated details
    let errors = [];

    if (!name || name.length < 2) {
        errors.push("Student Name should be at least 2 characters");
    }

    if (!age || age < 18) {
        errors.push("Student Age should be at least 18");
    }
    // Find student
    const student = students.find(s => s.sid === studentId);
    if (!student) {
        return res.send(`<h1>Error: Student with ID ${studentId} not found</h1>`);
    }

    // If there are errors reload the form with messages and previous data.
    if (errors.length > 0) {
        return res.render('editStudents', { 
            student: { sid: studentId, name, age }, 
            errors 
        });
    }

    // Update the student details
    student.name = name;
    student.age = age;

    // Redirect back to the students page
    res.redirect('/students');
});
module.exports = router;
