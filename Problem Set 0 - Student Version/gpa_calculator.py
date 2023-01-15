from typing import List
from college import Student, Course
import utils

def calculate_gpa(student: Student, courses: List[Course]) -> float:
    '''
    This function takes a student and a list of course
    It should compute the GPA for the student
    The GPA is the sum(hours of course * grade in course) / sum(hours of course)
    The grades come in the form: 'A+', 'A' and so on.
    But you can convert the grades to points using a static method in the course class
    To know how to use the Student and Course classes, see the file "college.py"  
    '''
    #TODO: ADD YOUR CODE HERE
    hours , points = 0, 0 
    for i in range(len(courses)):
        if student.id in courses[i].grades:
            hours += courses[i].hours 
            points += courses[i].hours * courses[i].convert_grade_to_points(courses[i].grades[student.id])
    return points/hours if hours else 0
