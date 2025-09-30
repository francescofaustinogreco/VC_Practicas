# Visual Computing - Práctica 2 Soluciones  

**Autores:** Francesco Faustino Greco - Bianca Cocci  
**GRUPO 05**

Este repositorio contiene las soluciones al segundo conjunto de ejercicios del curso de Visual Computing.  
Cada tarea fue resuelta utilizando **Python, OpenCV y NumPy**. A continuación se presenta una breve explicación de cómo se abordó cada ejercicio.  

## Ejercicios  

1. **Cuenta de píxeles blancos por filas**  
   Se implementó un conteo de píxeles blancos por filas (en lugar de columnas).  
   Se determinó el valor máximo de píxeles blancos por fila (`maxfil`), mostrando las filas y sus posiciones con un número de píxeles blancos mayor o igual que `0.90 * maxfil`.  

2. **Umbralizado y comparación Sobel vs. Canny**  
   Se aplicó un umbralizado a la imagen resultante de Sobel (convertida a 8 bits) y posteriormente se realizó el conteo de píxeles no nulos por filas y columnas, de manera similar al ejemplo con la salida de Canny.  
   Se calcularon los valores máximos por filas y columnas, destacando aquellas que superaban el `0.90 * máximo`.  
   Finalmente, se remarcaron dichas filas y columnas sobre la imagen del mandril y se compararon los resultados entre Sobel y Canny.  

3. **Demostrador interactivo con cámara**  
   Se propuso un demostrador que captura imágenes de la cámara y permite mostrar lo aprendido en estas prácticas a un público externo.  
   Además de mostrar la imagen original de la webcam, el sistema permite cambiar de modo, incluyendo al menos dos procesamientos distintos aplicados con funciones de OpenCV trabajadas hasta ahora.  

4. **Inspiración en proyectos artísticos**  
   Tras ver los vídeos *My little piece of privacy*, *Messa di voce* y *Virtual air guitar*, se diseñó un demostrador reinterpretando la parte de procesamiento de la imagen y sonido, aplicando técnicas vistas en clase para explorar la interacción creativa en tiempo real.  

## Notas  

Todas las soluciones fueron desarrolladas paso a paso durante la práctica.  
Cuando no sabíamos cómo proceder, buscamos pistas y ejemplos en StackOverflow o YouTube.  
El objetivo no fue la perfección, sino experimentar con OpenCV y la manipulación de imágenes.  

**Traducción realizada con ayuda de ChatGPT.**
