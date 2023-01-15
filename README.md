# Proyecto para el ramo de Inteligencia Artificial

### MNIST es una base de datos de dıgitos escritos a mano, que estan representados como una imagen de 28x28 pixeles en escala de grises donde cada pixel tiene una tonalidad que va desde 0 (blanco) a 1 (negro). La base de datos esta compuesta por 70.000 ejemplos (60.000 dispuestos para entrenamiento y 10.000 para pruebas). Estan disponibles dos archivos “Train.csv” y “Test.csv”, cada archivo compuesto por 60.000 y 10.000 lıneas, respectivamente. Cada lınea esta por 784 valores entre 0 y 255 que representan las distintas tonalidad de cada una de las 28 × 28 casillas, ademas de un valor 785 que indica el dıgito dibujado (entre 0 y 9).

## Se pide:

- Resolver el problema planteado utilizando Redes de tipo Perceptron MultiCapa (MLP). Puede usar como
  modelo los ejercicios vistos en clases.

## Defina:

- La arquitectura de la red (cuantas capas y neuronas tendra).
- Cual sera la funcion de activacion de cada neurona.
- Cual sera la funcion de error para la red.
- Cuantas iteraciones de los datos de entrenamiento aplicara a su red.

## Debe:

- probar distintos modelos de red, de manera de encontrar la configuracion que minimize su error,
  evitando el overfitting.

## Pruebas:

1.  Para comenzar planteamos la primera solución con una capa de entrada, una oculta y una última de salida. Como la capa oculta posee 30 neuronas, decidimos hacer 3 iteraciones, dando como resultado una precisión de 89% en la primera vuelta y 95% en la última.

2.  Luego probamos que pasaba con la precisión al momento de disminuir la cantidad de neuronas a solo 20 y que fueran solo 2 iteraciones, dándonos como resultado un peor rendimiento en la iteración final con un porcentaje del 93%

3.  Para aumentar ese porcentaje probamos aumentando las iteraciones a 5 y nos dio un porcentaje de precisión de 95% al igual que la primera vez.

4.  Así que decidimos mantener la cantidad de neuronas e iteraciones, pero agregando dos neuronas más a la prueba, dándonos como resultado un porcentaje igual del 95% que la vez anterior, por lo tanto, agregar capas no mejoro el porcentaje de precisión.

5.  Así que por ahora mantenemos las iteraciones, pero disminuimos la cantidad de capas a solo 1 pero con 50 neuronas, dándonos un porcentaje de 97%. Entonces nos planteamos la pregunta. ¿Es más importante la cantidad de neuronas que capas? Así que decidimos hacer más pruebas para averiguarlo.

6.  Aumentamos la cantidad de neuronas de la única capa oculta y mejoro el porcentaje a 98% de precisión, pero pareciera haber un pequeño overfitting.

7.  Para verificar el overfitting decidimos aumentar la cantidad de iteraciones a 10 pero manteniendo la cantidad de capas y neuronas. Dándonos como resultado un porcentaje de 99% de precisión

8.  Pero aún no logramos verificar si el overfitting es por la cantidad de épocas o por cantidad de neuronas. Así que decidimos seguir aumentando las vueltas pero ahora a 20. Demostrando que luego de la séptima iteración no hay una diferencia significativa en la precisión. Por lo que demasiadas vueltas dan overfitting

9.  Así que para intentar nivelar probamos con 2 capas ocultas de 50 neuronas cada una, dándonos un resultado similar a las anteriores, donde pareciera verse un overfitting en la cantidad de iteraciones, sobre todo después de la 7.ª iteración.

10. Así que decidimos aumentar la cantidad de neuronas, pero manteniendo las capas ocultas y ha mejorado desde un inicio el porcentaje de precisión. De momento pareciera que con un mayor número de neuronas se obtiene un mejor resultado que con un mayor número de capas ocultas.

11. Verificando la hipótesis anterior, deberíamos esperar que al aumentar el número de capas, pero disminuyendo el número de neuronas, la precisión de la red sea menor y efectivamente resulta cierto.

12. Así que para verificar aún más la hipótesis que pareciera ser cierta, añadimos el doble de capas ocultas. Dejando un total de 8 resultandos en que la precisión de la red es menor que la anterior.

13. Y para salir de todas las dudas, hicimos la prueba final que consistió en tener solo dos capas ocultas pero con 200 neuronas cada una. Dándonos como resultado un porcentaje de precisión de 99% que es el mejor resultado hasta el momento.

### _Como conclusión final pareciera ser evidente que es mejor tener una buena cantidad de neuronas por capas, que tener muchas capas y pocas neuronas._
