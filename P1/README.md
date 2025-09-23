Authors: Francesco Faustino Greco - Bianca Cocci GRUPO 05  

# Visual Computing - Practice 1 Solutions

This repository contains the solutions to the first set of exercises for the Visual Computing course.  
Each task was solved using Python, OpenCV, and NumPy. Below is a short explanation of how each exercise was approached.

---

## Exercises

### 1. Chessboard Image
We created an 800x800 grayscale image simulating a chessboard pattern.  
This was done by alternating white and black blocks of fixed size.

### 2. Mondrian-style Image
Using OpenCV drawing functions, we reproduced a simple composition inspired by Mondrian’s style.  
We combined rectangles and thick black lines with basic colors (red, yellow, blue).

### 3. Channel Modification
A live video frame was captured and one color channel (e.g. red) was modified freely.  
We experimented by enhancing the red channel and reducing the intensity of green and blue.

### 4. Brightest and Darkest Pixels
From an image the brightest and darkest pixels were found.  
We then drew colored circles on that positions.  
The same idea was later extended to detect the brightest/darkest 8x8 regions.

### 5. Pop Art Effect
We created a custom "pop art" composition by combining resized versions of the live video feed with different color modifications.  
The result was a 2x2 collage with playful, vivid variations.

---

## Notes
- All solutions were developed step by step during practice.  
- Whenever we did not know how to proceed, we looked for hints and examples on **StackOverflow** or **YouTube**.  
- The focus was not on perfection, but on experimenting with OpenCV and image manipulation.

---
