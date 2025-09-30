# Visual Computing - Práctica 2 Soluciones  

**Autores:** Francesco Faustino Greco - Bianca Cocci  
**GRUPO 05**

Este repositorio contiene las soluciones al segundo conjunto de ejercicios del curso de **Visual Computing**.  
El objetivo principal de estas prácticas fue familiarizarnos con técnicas fundamentales de procesamiento de imágenes y video utilizando **Python, OpenCV y NumPy**, explorando tanto los aspectos técnicos como creativos.  

---

## Índice  

1. [Introducción](#introducción)  
2. [Requisitos](#requisitos)  
3. [Estructura del repositorio](#estructura-del-repositorio)  
4. [Ejercicios](#ejercicios)  
   - [Ejercicio 1: Cuenta de píxeles blancos por filas](#1-cuenta-de-píxeles-blancos-por-filas)  
   - [Ejercicio 2: Umbralizado y comparación Sobel vs. Canny](#2-umbralizado-y-comparación-sobel-vs-canny)  
   - [Ejercicio 3: Demostrador interactivo con cámara](#3-demostrador-interactivo-con-cámara)  
   - [Ejercicio 4: Inspiración en proyectos artísticos](#4-inspiración-en-proyectos-artísticos)  
5. [Notas](#notas)  
6. [Créditos](#créditos)  

---

## Introducción  

La segunda práctica del curso se centra en el análisis de imágenes, el uso de filtros, el conteo de píxeles y la experimentación con efectos visuales en tiempo real.  
Además de la parte técnica, también se buscó un enfoque experimental y artístico, tomando inspiración en proyectos interactivos.  

---

## Requisitos  

Para ejecutar los notebooks de esta práctica, se recomienda tener instalados los siguientes paquetes de Python:  

- Python 3.8 o superior  
- OpenCV (cv2)  
- NumPy  
- Jupyter Notebook o JupyterLab  

La instalación puede hacerse fácilmente con:  

```bash
pip install opencv-python numpy jupyter
```

---

## Estructura del repositorio  

- `VC_P2.ipynb`: Notebook principal con la implementación de las soluciones.  
- `imagenes/`: Carpeta con las imágenes utilizadas en los ejercicios.  
- `README.md`: Documento de explicación y guía (este archivo).  

---

## Ejercicios  

### 1. Cuenta de píxeles blancos por filas  
Se implementó un conteo de píxeles blancos por filas (en lugar de columnas).  
Se determinó el valor máximo de píxeles blancos por fila (`maxfil`), mostrando las filas y sus posiciones con un número de píxeles blancos mayor o igual que `0.90 * maxfil`.  
Esto permitió identificar las zonas más iluminadas de la imagen con un criterio estadístico simple.  

---

### 2. Umbralizado y comparación Sobel vs. Canny  
Se aplicó un umbralizado a la imagen resultante de Sobel (convertida a 8 bits) y posteriormente se realizó el conteo de píxeles no nulos por filas y columnas, de manera similar al ejemplo con la salida de Canny.  
Se calcularon los valores máximos por filas y columnas, destacando aquellas que superaban el `0.90 * máximo`.  
Finalmente, se remarcaron dichas filas y columnas sobre la imagen del **mandril** y se compararon los resultados obtenidos con Sobel y Canny.  
Este ejercicio ayudó a comprender las diferencias prácticas entre los dos métodos de detección de bordes.  

---

### 3. Demostrador interactivo con cámara  
Se diseñó un demostrador que captura imágenes en tiempo real desde la cámara del ordenador.  
El sistema permite:  

- Mostrar la imagen original de la webcam.  
- Cambiar entre diferentes modos de visualización.  
- Aplicar al menos dos procesamientos distintos, implementados con funciones de OpenCV vistas en clase.  

Este ejercicio es clave para mostrar lo aprendido a un público externo, de manera práctica y visual.  

---

### 4. Inspiración en proyectos artísticos  
Tras visualizar los vídeos:  

- [My little piece of privacy](https://www.niklasroy.com/project/88/my-little-piece-of-privacy)  
- [Messa di voce](https://youtu.be/GfoqiyB1ndE?feature=shared)  
- [Virtual air guitar](https://youtu.be/FIAmyoEpV5c?feature=shared)  

Se diseñó un demostrador que reinterpretara el procesamiento de imagen y sonido desde una perspectiva creativa.  
El objetivo fue mostrar cómo las herramientas técnicas pueden servir de base para propuestas interactivas y artísticas.  

---

## Notas  

- Todas las soluciones fueron desarrolladas de manera progresiva durante la práctica.  
- Cuando no sabíamos cómo avanzar, buscamos inspiración en **StackOverflow, YouTube y documentación oficial de OpenCV**.  
- El objetivo principal no fue la perfección, sino la **exploración y experimentación** con técnicas de procesamiento de imágenes.  
- La práctica también fomentó la creatividad y la capacidad de aplicar conceptos técnicos a situaciones novedosas.  

---

## Créditos  

**Autores:** Francesco Faustino Greco - Bianca Cocci (GRUPO 05)  
**Traducción realizada con ayuda de Catgut.**  
