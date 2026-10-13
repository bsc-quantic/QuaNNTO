# Tablas cuánticas sin reentrenamiento

Este parche evalúa los **operadores ya guardados** de las arquitecturas de
`main_v1`, figuras 3 y 4. No llama a ningún entrenador, no modifica parámetros,
no escoge modelos por fidelidad y no deserializa modelos con pickle/jsonpickle.

## Instalación y ejecución

Descomprimir el ZIP de cambios **en la raíz del repositorio v2**, conservando las
rutas. Desde esa raíz, con Python 3.10–3.12, se recomienda un entorno independiente:

```bash
python -m venv .venv-quantum-evaluation
source .venv-quantum-evaluation/bin/activate
python -m pip install -r requirements-quantum-evaluation.txt
python -m quannto.tasks.quantum_tables --audit --output quantum_evaluation
python -m quannto.tasks.quantum_tables --task cat --output quantum_evaluation
```

La auditoría necesita solamente NumPy/SciPy. Devuelve código 2 cuando faltan
checkpoints: eso es lo esperado para la cúbica en el ZIP suministrado. La evaluación
de cats debe devolver código 0. Para evaluar ambos tasks cuando estén sus archivos:

```bash
python -m quannto.tasks.quantum_tables --output quantum_evaluation
```

Los resultados son `quantum_tables.csv`, `quantum_tables.json`, `cat_table.tex` y
`cubic_table.tex`. JSON incluye los puntos individuales, moments reales/imaginarios,
convergencia por cutoff, versiones, rutas y hashes SHA-256 de los archivos usados.
CSV separa el **mínimo histórico** y la **loss recalculada del checkpoint**. La tabla
LaTeX muestra la loss del checkpoint y su fidelidad; pone guiones cuando faltan datos
verificados. La fidelidad de un checkpoint cuya loss no coincide con el mínimo se
conserva en JSON/CSV con estado `loss_mismatch`, sin atribuirle ese mínimo histórico.

Se puede escoger un modelo, otra ubicación de datos o ampliar los cutoffs:

```bash
python -m quannto.tasks.quantum_tables --model cat_n2_aux_double --output quantum_evaluation
python -m quannto.tasks.quantum_tables --task cubic --cutoffs 30 40 60 80 --reference-cutoffs 200 300 --max-memory-mib 2048 --output quantum_evaluation
```

`--models-root` apunta a la carpeta que contiene `trained_operators`, `params`,
`train_losses` y `training_records`; `--datasets-root` apunta a los datasets.
También se respetan las ubicaciones habituales de QuaNNTO mediante sus variables
de entorno. El JSON `quannto/tasks/quantum_tables_manifest.json` fija de forma
explícita cada arquitectura y los archivos que le corresponden. Dentro del JSON,
los modos empiezan en **0**; las tablas para el paper los muestran desde **1**.

## Resultados verificados en el ZIP recibido

| Configuración | N | L | Sustracciones por capa (modos desde 1) | Loss del checkpoint | Fidelidad cuadrada | Cutoff final |
|---|---:|---:|---|---:|---:|---:|
| cat_n1_double | 1 | 1 | [[1, 1]] | 8.77402392629 | 0.982417998859 | 60 |
| cat_n2_aux_double | 2 | 1 | [[2, 2]] | 0.0622683432357 | 0.851380555145 | 40 |
| cat_n2_both | 2 | 1 | [[1, 2]] | 1.19830424253 | 0.0752959793618 | 60 |
| cat_n3_aux_pair | 3 | 1 | [[2, 3]] | 0.0613985080051 | 0.823538970304 | 40 |
| cat_n2_l2 | 2 | 2 | [[2], [2]] | 0.0672918213436 | 0.770079022472 | 40 |

Estos resultados corresponden al cat par de amplitud 1, input coherente -1,
15 momentos y estado reducido del modo 1. Las cinco losses recalculadas concuerdan
con los mínimos de los historiales vinculados, dentro de las tolerancias declaradas.
En ejecuciones antiguas sin registro de parámetros por iteración, esa concordancia
numérica no demuestra por sí sola una identidad histórica del checkpoint.

**No son cinco entrenamientos nuevos de cuatro saltos.** Solo `cat_n2_aux_double`
dispone aquí de un registro que confirma `basinhopping.niter=4`. Las otras filas usan
los operadores históricos disponibles; cambiar el script no cambia sus ejecuciones.

El caso de un modo ilustra que minimizar la loss de momentos no ordena los modelos
por fidelidad: su loss es mayor, aunque su fidelidad también lo es. Se calcula
siempre la fidelidad del mismo checkpoint cuya loss se reporta.

### Procedencia de los cats

- `cat_n1_double`: operadores con `rng-1to1`, loss `.txt` con
  `rng-1.0to-1.0`. El dataset tiene una única entrada -1; la loss física coincide.
- `cat_n2_aux_double`: **nueva** ejecución `rng-1to1`, con `training_record` y
  matrices/parámetros coherentes. El historial `.txt` bajo `rng-1.0to-1.0` coincide
  con el registro completo. El `.npy` de ese nombre es de la ejecución antigua
  (mínimo ≈0.859497), y no se mezcla con los operadores nuevos (mínimo ≈0.0622683).
- `cat_n2_both` y `cat_n3_aux_pair`: se usan explícitamente los pares `S_l,D_l`.
  Sus parámetros acompañantes no reconstruyen esas matrices con la parametrización
  actual. Se han seleccionado los operadores que reproducen la loss histórica;
  no se combinan ambos formatos. El manifiesto indica `checkpoint_source="operators"`.
- `cat_n2_l2`: pareja de operadores de 15 momentos, no la variante de 8 momentos.

La regla general `checkpoint_source="auto"` exige concordancia entre matrices y
parámetros si existen ambos. `"parameters"` reconstruye matrices usando la
parametrización actual de QuaNNTO sin entrenar. Las excepciones históricas anteriores
quedan explícitas; no se elige automáticamente otro archivo de nombre parecido.

### Archivos que faltan para la figura 3

En el ZIP **no están los checkpoints completos correspondientes a las cinco curvas**:

| ID | N | L | Sustracciones por capa (desde 0) | Mínimo de la loss guardada |
|---|---:|---:|---|---:|
| cubic_n2_l1 | 2 | 1 | `[[1]]` | 80.7701229524 |
| cubic_n2_l2 | 2 | 2 | `[[1],[1]]` | 9.21411143375 |
| cubic_n2_l3 | 2 | 3 | `[[1],[1],[1]]` | 14.1293494870 |
| cubic_n3_l1 | 3 | 1 | `[[1,2]]` | 9.28071230758 |
| cubic_n4_l1 | 4 | 1 | `[[1,2,3]]` | 3.65682576335 |

Para cada nombre `checkpoint_stem` indicado en el manifiesto, aportar una opción:

- `trained_operators/<checkpoint_stem>_S_l.npy` y
  `trained_operators/<checkpoint_stem>_D_l.npy`;
- `params/<checkpoint_stem>.npy` o `.txt`, si usa la parametrización actual.

El informe de auditoría escribe los nombres completos que faltan. Una lista finita
de momentos predichos y una curva de loss no bastan para recuperar un estado.

El notebook `nongauss_gate_synthesis_QONN.ipynb` contiene parámetros **redondeados**
impresos de dos ejecuciones: sus losses son 138.7765339548 y 11.4890584913, frente a
80.7701229524 y 9.2807123076 en los archivos de las curvas correspondientes. No se
presentan como los checkpoints de la figura 3. Los otros tres no aparecen allí.
Si se conservan los archivos originales, basta copiarlos y ejecutar el evaluador;
no hace falta entrenar. Este ZIP por sí solo no permite garantizar su recuperación.

## Definiciones y controles numéricos

La loss física del checkpoint es la usada en el entrenamiento:

\[
\mathcal L=\frac1M\sum_i\sum_{o\in\mathcal O}
\left|\langle o\rangle_{\mathrm{QONN},i}-m_{o,i}^{\mathrm{target}}\right|^2.
\]

Se suma sobre momentos y se promedia sobre muestras, incluyendo ambas componentes
complejas. Se leen los **targets guardados**, y se comprueba que correspondan a la
referencia declarada. Se conservan exactamente los 15 operadores de cats y los seis
de cúbica, en el orden del entrenador. Las matrices de los operadores se construyen
con margen para respetar su orden en el borde Fock: por ejemplo, $aa^\dagger=n+1$.
La loss histórica se lee directamente, sin aplicar `ln(1+loss)`.

Para cada input, se normaliza el estado condicional y se traza sobre los modos
auxiliares. La fidelidad es

\[
F=\langle\psi_{\mathrm{target}}|\rho_{\mathrm{out}}|\psi_{\mathrm{target}}\rangle.
\]

Para estados puros se reduce a $|\langle\psi|\phi\rangle|^2$. El target se calcula en
un espacio mayor; **no se renormaliza su proyección al cutoff menor del QONN**, pues
eso inflaría la fidelidad. Las normas asociadas a operadores ideales $a$ pueden
exceder uno y se etiquetan como `ideal_operator_weight`, no probabilidades físicas.

Los `S_l` actúan en orden de capa creciente, con orden xxpp. `D_l` contiene partes
reales e imaginarias de **amplitudes coherentes**, no medias de cuadraturas. El
estado gaussiano final se convierte a Fock con The Walrus, el conversor usado por
Strawberry Fields, sin renormalizar la truncación. Las sustracciones anteriores se
conjugan por todas las Gaussianas posteriores: para $G^\dagger bG=Bb+\beta$,
$GbG^\dagger=B^{-1}(b-\beta)$. Esto incluye todos los desplazamientos acumulados.

Se reserva un margen Fock igual al número total de sustracciones antes de aplicar
el polinomio y recortar. Así se obtienen los coeficientes proyectados del circuito
completo, sin introducir truncaciones en cada gate intermedia. No se reutiliza el
acumulador `D_concat` del entrenador histórico. Si los operadores recuperados de
una ejecución antigua no reproducen su loss registrada, el informe lo identifica;
no reemplaza esa loss ni modifica los parámetros silenciosamente.

Por defecto se prueban cutoffs **20, 30, 40, 60**, hasta lograr **dos comparaciones
consecutivas** que cumplan, en cada input:

- Cambio de fidelidad ≤$10^{-7}$.
- Cambio relativo de la norma proyectada ≤$10^{-8}$.
- Cambio de cada momento, dividido por $1+|m|$, ≤$10^{-7}$.

Se requiere también estabilidad del target entre cutoffs **200 y 300**, con
infidelidad ≤$10^{-8}$ y normas cercanas a uno. La comparación con la loss histórica
usa `atol=1e-6`, `rtol=1e-5`. Son comprobaciones numéricas de convergencia, no cotas
rigurosas del error en dimensión infinita. El límite de memoria predeterminado
estima seis tensores complejos de trabajo, hasta 512 MiB; excluye la sobrecarga JIT.

### Alcance de la fidelidad cúbica

Los targets existentes corresponden a **SF `Vgate(0.2)` con $\hbar=2$**,
$V=\exp[i\gamma x^3/(3\hbar)]$. En las coordenadas $\hbar=1$ del manuscrito,
$\exp[i\gamma_{\mathrm{paper}}x^3/3]$ tiene
$\gamma_{\mathrm{paper}}=0.2\sqrt2$. No se retargetean los modelos a otro operador.
La definición se comprueba contra las 50 muestras guardadas.

La loss se evalúa en esas 50 amplitudes de entrenamiento. Para fidelidad se usan
los **49 puntos medios** entre amplitudes consecutivas, separados del entrenamiento.
Se reportan media aritmética, mínimo y amplitud donde aparece ese mínimo, sin
seleccionar checkpoints por estos resultados. Los endpoints pertenecen al training.

Esto es fidelidad del **estado de salida condicionado** sobre una familia concreta
de coherentes reales. No es fidelidad de proceso, de Choi ni fidelidad promedio de
una puerta sobre todos los estados. Debe describirse así en la respuesta al referee.

Referencias de convenciones y backends:
[GaussianTransform](https://strawberryfields.readthedocs.io/en/latest/code/api/strawberryfields.ops.GaussianTransform.html),
[Vgate](https://strawberryfields.readthedocs.io/en/latest/code/api/strawberryfields.ops.Vgate.html).

## Verificación

Validación de esta entrega: **25 pruebas pasadas en el entorno Fock**, más
**una prueba adicional pasada con JAX 0.6.2** que compara el decodificador con
`QNN.build_QNN` en tres arquitecturas. No se ha reentrenado ningún modelo.

```bash
python -m pip install pytest
python -m pytest test/test_quantum_tables.py
```

Las pruebas cubren fidelidad cuadrada e invariancia bajo fase global; reducción de
estados entrelazados; colas del target; coherentes con cutoff alto; orden de momentos;
lectura NPY/texto; rechazo de formatos inconsistentes; circuitos de una, dos y tres
capas contra Strawberry Fields directo; las cinco parejas reales de operadores cat;
y exportación de tablas con un circuito cúbico de referencia conocido.

Para este último, las 99 fidelidades (50 training + 49 evaluación) se contrastan con
una integral independiente de la función de onda en posición, mediante cuadratura
de Gauss–Hermite. Se comprueba también que un gamma incorrecto sea rechazado. Las
99 referencias cúbicas pasan el contraste de cutoffs 200/300 (máxima infidelidad
observada ≈$3.61\times10^{-13}$).

La prueba adicional del decodificador compara con `QNN.build_QNN` actual; requiere
JAX y las dependencias normales del repo. Se omite en el entorno Fock independiente,
que no necesita JAX para ejecutar las tablas. Ninguna prueba optimiza parámetros.

## Ficheros del parche

| Fichero | Cambio |
|---|---|
| `quannto/utils/strawberryfields/general_tools.py` | Fidelidad puro–puro cuadrada; validación de estados; coherentes estables a cutoff alto; momentos sin matrices gigantes del entorno; imports SF locales. |
| `quannto/utils/strawberryfields/qonn_builder.py` | Corrección de documentación de unidades y de normas ideales; conserva el simulador directo para contraste. |
| `quannto/utils/strawberryfields/quantum_state_evaluation.py` | Nuevo evaluador físico, referencias y decodificación de parámetros. |
| `quannto/tasks/quantum_tables.py` | Nueva CLI de auditoría, convergencia y exportación. |
| `quannto/tasks/quantum_tables_manifest.json` | Mapeo explícito de las diez arquitecturas y sus archivos. |
| `requirements-quantum-evaluation.txt` | Entorno Fock separado y compatible. |
| `test/test_quantum_tables.py` | Pruebas científicas y de procedencia. |
| `QUANTUM_TABLES.md` | Estas instrucciones, resultados y límites. |

Los datasets, modelos, parámetros, matrices, historiales, entrenadores y manuscrito
originales no se sobrescriben. El ZIP contiene únicamente estos ocho archivos,
comparados con `quannto-master-100926v2(1).zip` (idéntico a `quannto-master-100926v2.zip`).
