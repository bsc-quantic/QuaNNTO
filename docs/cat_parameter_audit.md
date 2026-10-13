# Auditoría de las ramas casi impares del benchmark cat-state

Se analiza el repo `quannto-master-011026.zip`, sin modificar los modelos originales.
Objetivo: even cat de amplitud 1; entrada coherente -1; pérdida SSE no ponderada sobre los 15 momentos originales (hasta grado 6). Las fidelidades usan la convención de solapamiento cuadrado. Los índices K de este documento son 1-based; los del código son 0-based.

## Resultado principal

La pérdida y la fidelidad casi nula se reproducen propagando los parámetros entrenados con el simulador independiente de Fock incluido en el repo. No son un desacuerdo entre los resultados guardados y esta reconstrucción. Para N=1,L=1,K=(1), el estado impar constituye un mínimo local numérico también al incluir desplazamiento complejo y orientación del squeezing. Para N=2,L=1,K=(1,2) y N=1,L=2,K=(1);(1), no se ha encontrado una dirección de escape robusta con estos diagnósticos, pero hay direcciones casi planas y no se certifica un mínimo estricto ni global.

| Arquitectura | Pérdida reconstruida | F(even) | F(odd, alpha=1) | Paridad |
|---|---:|---:|---:|---:|
| N=1,L=1,K=(1) | 2.329901895824 | 9.12e-15 | 0.9908950571 | -1 |
| N=2,L=1,K=(1,2) | 2.329901895825 | 2.99e-13 | 0.9908950557 | -1 |
| N=1,L=2,K=(1);(1) | 2.270356406496 | 0.0003223001 | 0.9905635512 | -0.9990669265 |

Las pérdidas con cutoffs 60 y 80 difieren menos de 2e-13 en los tres checkpoints. El cutoff 40 todavía introduce ~4.3e-6 de error en la pérdida de dos capas; por ello se utilizó 60 para derivadas y 80 para verificar los estados refinados. La pequeña cola de la salida no es por sí sola una garantía de convergencia de la propagación intermedia; la comparación de cutoffs es el control usado aquí.

## Qué se ha calculado

1. Reconstrucción a partir de los archivos `.npy` de parámetros, con `parameter_fock_reference.py`: fases, interferómetros, squeezing, desplazamientos y sustracciones en su orden físico. Se respeta el orden invertido de bloques de capas y el valor absoluto de los parámetros de squeezing del entrenamiento.
2. Pérdida contra el array de targets original, no contra un target ajustado para el diagnóstico.
3. Gradientes y Hessianos en los 5, 14 y 10 parámetros, respectivamente, con diferencias centrales de pasos 2e-4 y 5e-5. Se añade extrapolación de Richardson para cancelar el error principal O(h²).
4. Exploración de la dirección de menor curvatura y refinamientos L-BFGS-B con las cotas originales. El caso de dos capas se refina también por mínimos cuadrados con Jacobiano de diferencias centrales.
5. Comprobación separada de las cuatro coordenadas físicas de la familia monomodo de una capa.

Estos cálculos no ejecutan JAX ni vuelven a entrenar todos los modelos del paper. Evalúan la misma función objetivo mediante propagación Fock convergida en los puntos examinados. Tampoco prueban el comportamiento del Hessiano completo para los grados 8–12; el barrido anterior de esos grados era bidimensional.

## Una capa y un modo: mínimo local fuera del corte real

Una transformación gaussiana monomodo aplicada a una entrada coherente fija produce un estado gaussiano puro arbitrario. Antes de la sustracción, sus cuatro coordenadas físicas pueden escribirse como beta compleja, r y theta:

`psi ∝ a D(beta) R(theta) S(-r)|0>`.

Se usa `R(theta)=exp(-i theta N)` y la convención `S(r)=exp[r(a²-a†²)/2]`. El mínimo estudiado está en beta=0, theta=0, r=0.2508015053; su salida es proporcional a S(-r)|1>.

En las coordenadas (Re beta, Im beta, r, theta), el Hessiano extrapolado tiene autovalores aproximadamente:

`247.36, 4332.08, 13808.57, 33541.09`.

Todos son claramente positivos. El quinto parámetro del circuito es redundante para el estado preparado a partir de una entrada fija; no debe interpretarse su curvatura casi nula como una dirección física de escape. Este resultado es evidencia numérica de un mínimo local de la familia monomodo completa, no solo del corte beta,r reales. No certifica el mínimo global de esa familia ni imposibilidad de obtener mayor fidelidad con otros parámetros.

## Dos modos y una capa

El Hessiano extrapolado presenta cinco autovalores positivos bien separados de cero, aproximadamente:

`12.36, 317.29, 3914.32, 28981.61, 51476.26`.

Los nueve restantes tienen magnitudes inferiores a ~1e-4. Es incorrecto declarar un saddle a partir del pequeño autovalor negativo (~-3.2e-4) obtenido con el paso más grande: no es estable al reducir el paso. La refinación desde el checkpoint mantiene pérdida 2.329901895823 y F(even) ~1e-13.

La salida medida tiene pureza prácticamente 1 y coincide con la rama squeezed-one. Eso no significa que toda esta arquitectura esté restringida a estados impares ni que todas las direcciones planas sean redundancias. El Hessiano semidefinido no excluye escape a órdenes superiores ni soluciones mejores lejos del checkpoint.

## Un modo y dos capas

El checkpoint original aún presenta un gradiente extrapolado ~8.3e-3 en estas coordenadas, con curvaturas grandes y pequeñas mezcladas. Los autovalores negativos pequeños de ese punto no bastan para clasificarlo como saddle de un punto estacionario.

Los refinamientos desde el checkpoint y desde ambos lados de la dirección de menor autovalor alcanzan pérdidas cercanas a 2.270356405745: mejora de ~7.5e-10. F(even) permanece ~3.22e-4 y F(odd) ~0.990564. Los nuevos parámetros están solo en los JSON de diagnóstico; no se reemplazaron checkpoints.

Una refinación adicional de mínimos cuadrados reduce la norma del gradiente de la función SSE/2 a ~2.3e-5. Después, seis curvaturas positivas están bien separadas de cero (aprox. 1.03,3.84,872.84,22578.08,32795.80,125305.93); las otras cuatro son de orden 1e-5–1e-4. No se certifica su signo. El estado es casi impar, no exactamente un autovector de paridad.

## Interpretación científica

Los momentos sí distinguen los cats ideales: para alpha=1, <N>even=tanh(1) y <N>odd=coth(1). Por tanto, el diagnóstico no establece una indistinguibilidad even/odd de la loss. Establece que una función objetivo con información discriminante puede tener mínimos locales que producen un estado ortogonal, o casi ortogonal, al target.

Para el paper, una formulación respaldada por estos resultados sería:

> For the single-mode, single-layer architecture with one photon subtraction, the moment-matching objective exhibits an opposite-parity local minimum close to a squeezed single-photon state. This minimum persists when complex displacement and squeezing orientation are included. The other two near-zero-fidelity checkpoints remain close to the same odd-cat-like region under local refinement, although near-flat directions prevent a strict local-minimum classification from the present numerical Hessians.

No se puede generalizar esta afirmación a todas las arquitecturas, a todos los órdenes de momentos ni a mínimos globales. Los registros originales usan cuatro pasos de basinhopping y diferencias finitas en L-BFGS-B; esa exploración no aporta una garantía global.

La siguiente pregunta científica sería si reinicios que salgan de esta región encuentran mejores soluciones con la loss original. Cambiar la loss para incluir paridad o fidelidad sería un experimento distinto y debe identificarse como tal. En esta auditoría no se ha cambiado la loss de entrenamiento.

## Reproducción

Dependencias: Python, NumPy y SciPy. Descomprimir los tres scripts en la raíz de QuaNNTO, manteniendo el módulo de referencia del repo adjunto. Ejecutar:

```bash
OPENBLAS_NUM_THREADS=1 python -m quannto.tasks.diagnostics.catstate_full_parameter_audit --repo . --out results/local/cat_parameter_audit --refine
OPENBLAS_NUM_THREADS=1 python -m quannto.tasks.diagnostics.single_mode_physical_check --repo . --out results/local/cat_parameter_audit
OPENBLAS_NUM_THREADS=1 python -m quannto.tasks.diagnostics.refine_check --repo . --out results/local/cat_parameter_audit
```

El directorio `results/` de este paquete contiene los resultados ya calculados. `audit_terminal.txt`, `physical_terminal.txt` y `refine_terminal.txt` contienen las salidas de ejecución. Los valores extrapolados del primer diagnóstico se añadieron al JSON a partir de los mismos Hessianos guardados. Los scripts se adaptaron a argumentos --repo/--out tras la ejecución inicial; el cálculo numérico no cambió.

`branch_norm` es el peso de la rama para operadores ideales de sustracción; no es una probabilidad física de clic sin especificar el modelo de heralding.
