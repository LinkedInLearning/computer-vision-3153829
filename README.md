# Computer Vision on the Raspberry Pi 4
This is the repository for the LinkedIn Learning course Computer Vision on the Raspberry Pi 4. The full course is available from [LinkedIn Learning][lil-course-url].

![Computer Vision on the Raspberry Pi 4][lil-thumbnail-url] 

More and more applications are using computer vision to detect and recognize objects. These applications usually execute on large computers, but developers can save money and power by running them on single-board computers (SBCs). The Raspberry Pi 4 is one of the most popular SBCs available. It's also the first computer in the Raspberry Pi family powerful enough to execute computer vision applications. Also, the software needed to build these applications can be downloaded freely from the Internet.  In this course, instructor Matt Scarpino shows programmers how to write and execute computer vision applications on the Raspberry Pi 4. Matt introduces you to using the Thonny IDE, the OpenCV library, and NumPy array operations. He steps through object detection and neural networks, then explores convolutional neural networks (CNNs), including the Keras package and the TensorFlow package. Matt also walks you through what you can do with a Raspberry Pi HQ camera.

## Instructions
This repository has branches for each of the videos in the course. You can use the branch pop up menu in github to switch to a specific branch and take a look at the course at that stage, or you can add `/tree/BRANCH_NAME` to the URL to go to the branch you want to access.

## Branches
The branches are structured to correspond to the videos in the course. The naming convention is `CHAPTER#_MOVIE#`. As an example, the branch named `02_03` corresponds to the second chapter and the third video in that chapter. 
Some branches will have a beginning and an end state. These are marked with the letters `b` for "beginning" and `e` for "end". The `b` branch contains the code as it is at the beginning of the movie. The `e` branch contains the code as it is at the end of the movie. The `main` branch holds the final state of the code when in the course.

When switching from one exercise files branch to the next after making changes to the files, you may get a message like this:

    error: Your local changes to the following files would be overwritten by checkout:        [files]
    Please commit your changes or stash them before you switch branches.
    Aborting

To resolve this issue:
	
    Add changes to git using this command: git add .
	Commit changes using this command: git commit -m "some message"

## Installing
1. To use these exercise files, you must have the following installed:
	- [list of requirements for course]
2. Clone this repository into your local machine using the terminal (Mac), CMD (Windows), or a GUI tool like SourceTree.
3. [Course-specific instructions]


### Instructor

Matt Scarpino 
                            


                            

Check out my other courses on [LinkedIn Learning](https://www.linkedin.com/learning/instructors/matt-scarpino).

[lil-course-url]: https://www.linkedin.com/learning/computer-vision-on-the-raspberry-pi-4
[lil-thumbnail-url]: https://cdn.lynda.com/course/3153829/3153829-1635434273646-16x9.jpg
