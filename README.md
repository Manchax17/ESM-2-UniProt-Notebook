Contenido complementario
# TRANSFORMERS
Los transformers son un tipo de arquitectura de redes neuronales que transforma o cambia una secuencia de entrada en una secuencia de salida. Hacen esto aprendiendo contexto, es decir, haciendo el seguimiento de las relaciones entre secuencia de componentes. 
Ejemplo:
¿De qué color es el cielo?
El modelo de las redes transformers usan una representación matemática interna que identifica la relevancia y las relaciones entre las palabras "color", "cielo" y "azul". Este conocimiento hace que por salida sea:
"El cielo es azul"

## ¿Qué son?

Los **Transformers** son una arquitectura de red neuronal introducida en 2017 por investigadores de Google en el paper _"Attention Is All You Need"_ (Vaswani et al., 2017). Revolucionaron el campo del procesamiento de lenguaje natural (NLP) y hoy son la base de modelos como **GPT**, **BERT**, **Claude** y muchos otros.

A diferencia de arquitecturas anteriores como las RNN (Redes Neuronales Recurrentes) o las LSTM, los Transformers **no procesan las secuencias de manera paso a paso**, sino que analizan toda la secuencia **de forma paralela**, lo que los hace mucho más eficientes y potentes.

---

## Contexto Histórico

| Época         | Arquitectura     | Limitación                                                |
| ------------- | ---------------- | --------------------------------------------------------- |
| Antes de 2017 | RNN / LSTM       | Procesamiento secuencial, lento, olvido de contexto largo |
| 2017          | **Transformer**  | Supera las limitaciones anteriores con atención           |
| 2018+         | BERT, GPT, T5... | Modelos pre-entrenados masivos basados en Transformers    |

---

## Componentes Principales

### 1. 🔡 Embeddings de entrada

Antes de entrar a la red, cada token (palabra o subpalabra) se convierte en un **vector numérico** que representa su significado semántico.

### 2. 📍 Positional Encoding (Codificación Posicional)

Como el Transformer procesa todo en paralelo, necesita saber el **orden** de las palabras. Para esto, se le añade al embedding una señal matemática que indica la posición de cada token en la secuencia.

### 3. 🎯 Mecanismo de Atención (Self-Attention)

Es el corazón del Transformer. Permite que cada palabra de la secuencia "preste atención" a todas las demás palabras para entender mejor su contexto.

Para cada token, se calculan tres vectores:
La fórmula es:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
### Qué significa cada parte

- **Q (Query)** — la "pregunta" que hace cada palabra: _¿a qué otras palabras debo prestarle atención?_
- **K (Key)** — la "etiqueta" de cada palabra: _¿qué información ofrezco yo?_
- **V (Value)** — el contenido real de cada palabra: _si me prestas atención, esto es lo que te doy_
- **d_k** — la dimensión de los vectores K, se usa para escalar y evitar que los números sean demasiado grandes

### Cómo funciona paso a paso

1. **QKᵀ** — multiplica Queries por Keys, obteniendo un puntaje de "cuánto se relaciona cada palabra con cada otra"
2. **÷ √d_k** — divide para estabilizar los gradientes durante el entrenamiento
3. **softmax(...)** — convierte los puntajes en probabilidades (todos suman 1)
4. **× V** — usa esas probabilidades para hacer una suma ponderada de los Values

### En términos simples

Imagina la frase _"El gato que estaba en el tejado cayó"_. Cuando el modelo procesa _"cayó"_, la atención le permite saber que se refiere al _"gato"_ y no al _"tejado"_, porque los puntajes QKᵀ serán más altos entre esas palabras.


### 4. 🧠 Multi-Head Attention (Atención Multicabeza)

En lugar de calcular la atención una sola vez, se calculan **múltiples atenciones en paralelo** (cabezas), cada una aprendiendo distintos tipos de relaciones entre palabras. Los resultados se concatenan y se proyectan.

### 5. 🔁 Feed-Forward Network

Después de la atención, cada posición pasa por una **red neuronal feed-forward** (completamente conectada), aplicada de manera independiente a cada token.

### 6. ➕ Conexiones Residuales y Normalización

Cada subcapa (atención + feed-forward) incluye:

- **Conexión residual:** Se suma la entrada con la salida (`x + SubLayer(x)`)
- **Layer Normalization:** Estabiliza el entrenamiento

### 7. 📦 Encoder y Decoder

La arquitectura original tiene dos bloques:

| Bloque                | Función                                     | Usado en                  |
| --------------------- | ------------------------------------------- | ------------------------- |
| **Encoder**           | Comprende la secuencia de entrada           | BERT                      |
| **Decoder**           | Genera la secuencia de salida token a token | GPT                       |
| **Encoder + Decoder** | Traduce/transforma secuencias               | T5, modelos de traducción |

---

## Características Clave

- ✅ **Paralelismo:** Procesa todos los tokens simultáneamente, no en secuencia.
- ✅ **Contexto global:** Cada token puede relacionarse con cualquier otro token, sin importar la distancia.
- ✅ **Escalabilidad:** A mayor número de parámetros y datos, mejor rendimiento.
- ✅ **Versatilidad:** Funciona en texto, imágenes, audio, video y más.
- ✅ **Transfer Learning:** Se pre-entrena en grandes corpus y luego se ajusta a tareas específicas (_fine-tuning_).

---

## Propiedades Importantes

### 🔍 Atención es O(n²)

La complejidad del mecanismo de atención crece cuadráticamente con la longitud de la secuencia. Esto es una limitación para secuencias muy largas, lo que ha motivado variantes como _Longformer_, _FlashAttention_, etc.

### 📚 Pre-entrenamiento + Fine-tuning

Los Transformers se entrenan primero de forma **no supervisada** sobre enormes cantidades de texto, aprendiendo representaciones generales del lenguaje. Luego se ajustan a tareas específicas con mucho menos datos.

### 🌐 Generalización multi-dominio

Una vez pre-entrenados, los Transformers pueden adaptarse a tareas muy diversas: clasificación, traducción, resumen, generación de código, respuesta a preguntas, entre otras.

---

## Modelos Destacados basados en Transformers

| Modelo     | Empresa   | Tipo            | Aplicación            |
| ---------- | --------- | --------------- | --------------------- |
| **BERT**   | Google    | Solo Encoder    | Comprensión de texto  |
| **GPT-4**  | OpenAI    | Solo Decoder    | Generación de texto   |
| **T5**     | Google    | Encoder-Decoder | Traducción, resumen   |
| **Claude** | Anthropic | Solo Decoder    | Asistente IA          |
| **ViT**    | Google    | Solo Encoder    | Visión por computador |

---


## Resumen Visual del Flujo

```
Texto de entrada
     ↓
Tokenización + Embeddings
     ↓
Positional Encoding
     ↓
┌─────────────────────┐
│  Bloque Transformer │  ← Se repite N veces
│  ┌───────────────┐  │
│  │ Multi-Head    │  │
│  │ Self-Attention│  │
│  └───────┬───────┘  │
│          ↓ + residual│
│  ┌───────────────┐  │
│  │ Feed-Forward  │  │
│  │ Network       │  │
│  └───────┬───────┘  │
│          ↓ + residual│
└──────────┼──────────┘
           ↓
      Salida final
```

---

> 💡 **Idea clave para recordar:** El poder de los Transformers radica en el mecanismo de _atención_, que permite capturar relaciones complejas entre elementos de una secuencia sin importar cuán lejos estén unos de otros.

# El Mecanismo de Atención en Transformers 🎯

## ¿Por qué se necesita la "Atención"?

Imagina la frase:

> _"El banco estaba lleno de gente porque era día de pago"_

Para entender a qué se refiere **"banco"** (¿institución financiera? ¿asiento?), una persona necesita leer **toda la oración**. Los modelos anteriores (RNN/LSTM) tenían que "recordar" el contexto pasando información de token en token — como el teléfono roto — y perdían contexto en oraciones largas.

La **atención** resuelve esto: le permite a cada palabra "mirar" directamente a todas las demás y decidir cuáles son relevantes para su interpretación.

---

## La Intuición: Queries, Keys y Values

El mecanismo de atención se inspira en un sistema de **búsqueda de información**. Piénsalo así:

> Imagina que buscas un video en YouTube.
> 
> - Tu **búsqueda escrita** → **Query (Q)**
> - Los **títulos de los videos** → **Keys (K)**
> - El **contenido real del video** → **Values (V)**
> 
> YouTube compara tu Query con cada Key para saber qué tan relevante es cada video, y te devuelve el contenido (Value) de los más relevantes.

En un Transformer, **cada token** genera sus propios vectores Q, K y V a partir de sus embeddings, y el proceso ocurre para todos los tokens simultáneamente.

---

## Paso a Paso: ¿Cómo se calcula la Atención?

Supongamos que tenemos la frase con 4 tokens: `["El", "gato", "come", "pescado"]`

### Paso 1 — Obtener los vectores Q, K, V

Cada token tiene un **embedding** (vector de números). A partir de él, se calculan Q, K y V multiplicando por matrices de pesos aprendibles:

```
Para cada token xᵢ:

  Q_i = xᵢ · W_Q    ← "¿Qué estoy buscando?"
  K_i = xᵢ · W_K    ← "¿Qué puedo ofrecer?"
  V_i = xᵢ · W_V    ← "¿Qué información transmito?"
```

Donde `W_Q`, `W_K` y `W_V` son matrices que la red **aprende durante el entrenamiento**.

---

### Paso 2 — Calcular los Scores (puntuaciones)

Para saber cuánto debe "atender" el token `i` al token `j`, se calcula el **producto punto** entre sus vectores:

```
score(i, j) = Q_i · K_j
```

Ejemplo para el token **"gato"** mirando a todos los tokens:

```
score("gato", "El")      = Q_gato · K_El
score("gato", "gato")    = Q_gato · K_gato
score("gato", "come")    = Q_gato · K_come
score("gato", "pescado") = Q_gato · K_pescado
```

Un score alto significa que esos dos tokens están muy relacionados en este contexto.

---

### Paso 3 — Escalar los Scores

Los scores pueden volverse muy grandes con vectores de alta dimensión, lo que causa problemas en el gradiente. Para evitarlo, se dividen por la raíz cuadrada de la dimensión de los Keys `d_k`:

```
score_escalado(i, j) = (Q_i · K_j) / √d_k
```

> 💡 Si `d_k = 64`, se divide entre `√64 = 8`. Esto mantiene los valores en una escala estable.

---

### Paso 4 — Softmax: convertir scores en probabilidades

Se aplica **Softmax** a los scores escalados para que sumen 1 y representen "cuánta atención" poner en cada token:

```
α(i, j) = softmax(score_escalado(i, j))
```

Para el token **"gato"**, podría verse así (ejemplo ilustrativo):

```
α("gato", "El")      = 0.05   ← Poca atención
α("gato", "gato")    = 0.10   ← Algo de atención
α("gato", "come")    = 0.70   ← Mucha atención (el gato COME)
α("gato", "pescado") = 0.15   ← Algo de atención
                       ------
                       1.00   ✅
```

---

### Paso 5 — Calcular la salida con los Values

La salida para el token `i` es la **suma ponderada** de todos los Values, usando las puntuaciones de atención como pesos:

```
Output_i = Σ α(i, j) · V_j
           j
```

Para "gato":

```
Output_gato = 0.05·V_El + 0.10·V_gato + 0.70·V_come + 0.15·V_pescado
```

Este vector resultante es la **nueva representación** del token "gato", enriquecida con el contexto de toda la oración.

---

## La Fórmula Completa

Reuniendo todo en una sola expresión matricial (para toda la secuencia a la vez):

Attention(Q,K,V)=softmax(QKTdk)V\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)VAttention(Q,K,V)=softmax(dk​​QKT​)V

| Símbolo | Significado       | Dimensión típica        |
| ------- | ----------------- | ----------------------- |
| `Q`     | Matriz de Queries | `(n_tokens × d_k)`      |
| `K`     | Matriz de Keys    | `(n_tokens × d_k)`      |
| `V`     | Matriz de Values  | `(n_tokens × d_v)`      |
| `d_k`   | Dimensión de Keys | Ej: 64                  |
| `QKᵀ`   | Matriz de scores  | `(n_tokens × n_tokens)` |

---

## Multi-Head Attention: Múltiples perspectivas

Una sola atención puede capturar un tipo de relación. Pero el lenguaje es complejo: una palabra puede relacionarse con otras por **sintaxis**, **semántica**, **correferencia**, etc.

La solución: calcular la atención **h veces en paralelo**, cada una con sus propias matrices `W_Q`, `W_K`, `W_V`. Cada "cabeza" aprende relaciones distintas.

```
head_1 = Attention(Q·W_Q1, K·W_K1, V·W_V1)  ← Ej: relaciones sintácticas
head_2 = Attention(Q·W_Q2, K·W_K2, V·W_V2)  ← Ej: relaciones semánticas
...
head_h = Attention(Q·W_Qh, K·W_Kh, V·W_Vh)  ← Ej: correferencias
```

Las salidas se **concatenan** y se proyectan:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
```

> En el paper original se usan **8 cabezas** con `d_k = 64` cada una (dimensión total del modelo = 512).

---

## Visualización de la Atención

Cada celda de la matriz de atención indica cuánto atiende una palabra a otra:

```
              El   gato  come  pescado
          ┌─────┬─────┬──────┬────────┐
El        │0.60 │0.20 │ 0.10 │  0.10  │
gato      │0.05 │0.10 │ 0.70 │  0.15  │
come      │0.05 │0.60 │ 0.15 │  0.20  │
pescado   │0.05 │0.15 │ 0.60 │  0.20  │
          └─────┴─────┴──────┴────────┘

(Valores ilustrativos)
```

Nota cómo **"gato"** atiende fuertemente a **"come"**, y **"come"** atiende a **"gato"** — la red está capturando la relación sujeto-verbo.

---

## Tipos de Atención en un Transformer

| Tipo                      | Dónde ocurre | Descripción                                                                                         |
| ------------------------- | ------------ | --------------------------------------------------------------------------------------------------- |
| **Self-Attention**        | Encoder      | Cada token atiende a todos los tokens de la misma secuencia                                         |
| **Masked Self-Attention** | Decoder      | Como self-attention pero los tokens futuros se enmascaran (el modelo no puede "ver hacia adelante") |
| **Cross-Attention**       | Decoder      | Los tokens de salida atienden a los tokens de entrada (del Encoder)                                 |

---

## Complejidad Computacional

| Aspecto                 | Detalle                                                     |
| ----------------------- | ----------------------------------------------------------- |
| **Complejidad**         | O(n²·d) — cuadrática en longitud de secuencia               |
| **Paralelismo**         | Total — todos los scores se calculan a la vez               |
| **Limitación**          | Secuencias muy largas son costosas (ej: documentos enteros) |
| **Soluciones modernas** | FlashAttention, Sparse Attention, Longformer                |

---

## Resumen del Flujo Completo

```
Tokens: ["El", "gato", "come", "pescado"]
          ↓
    Embeddings (xᵢ)
          ↓
   ┌──────────────────────────────┐
   │  Multiplicar por W_Q, W_K, W_V│
   └──────────────┬───────────────┘
                  ↓
         Calcular QKᵀ (scores)
                  ↓
         Escalar por 1/√d_k
                  ↓
           Aplicar Softmax
                  ↓
      Ponderar los Values (·V)
                  ↓
   Nuevas representaciones contextuales
```

---

> 💡 **Idea para llevarte a clase:** La atención no es magia — es álgebra lineal. Tres matrices aprendibles (W_Q, W_K, W_V) transforman los embeddings en vectores que permiten calcular compatibilidades entre tokens. El resultado es que cada token obtiene una nueva representación que **incorpora el contexto de toda la secuencia**.






# Transformers: Ejemplos Matemáticos Paso a Paso 🧮

> Usaremos una frase corta a lo largo de todo el documento para que puedas seguir el hilo:
> 
> **"El gato duerme"** → 3 tokens, embeddings de dimensión 4 (en la práctica son 512 o más, pero con 4 es suficiente para entender).

---

## 📌 Punto de partida: Los Embeddings

Cada token se representa como un vector de números. Estos valores son **aprendidos durante el entrenamiento**. Para nuestro ejemplo:

```
x_El     = [1.0,  0.5,  0.2,  0.8]
x_gato   = [0.3,  1.0,  0.7,  0.1]
x_duerme = [0.6,  0.2,  1.0,  0.4]
```

Agrupados en una matriz X (3 tokens × 4 dimensiones):

```
X = | 1.0  0.5  0.2  0.8 |   ← "El"
    | 0.3  1.0  0.7  0.1 |   ← "gato"
    | 0.6  0.2  1.0  0.4 |   ← "duerme"
```

---

## 📍 Paso 1 — Positional Encoding

Como el Transformer procesa todo en paralelo, necesita saber el **orden** de los tokens. Se añade una señal de posición al embedding usando funciones seno y coseno:

PE(pos,2i)=sin⁡(pos100002i/d)PE(pos,2i+1)=cos⁡(pos100002i/d)PE(pos,2i)​=sin(100002i/dpos​)PE(pos,2i+1)​=cos(100002i/dpos​)

Con `d=4` (dimensión del embedding) y posiciones 0, 1, 2:

```
PE_0 ("El")     = [sin(0/1),   cos(0/1),   sin(0/100),   cos(0/100)  ]
                = [0.000,      1.000,       0.000,         1.000      ]

PE_1 ("gato")   = [sin(1/1),   cos(1/1),   sin(1/100),   cos(1/100)  ]
                = [0.841,      0.540,       0.010,         0.999      ]

PE_2 ("duerme") = [sin(2/1),   cos(2/1),   sin(2/100),   cos(2/100)  ]
                = [0.909,     -0.416,       0.020,         0.999      ]
```

Se **suman** al embedding original:

```
X' = X + PE

X'_El     = [1.0+0.000, 0.5+1.000, 0.2+0.000, 0.8+1.000] = [1.000, 1.500, 0.200, 1.800]
X'_gato   = [0.3+0.841, 1.0+0.540, 0.7+0.010, 0.1+0.999] = [1.141, 1.540, 0.710, 1.099]
X'_duerme = [0.6+0.909, 0.2-0.416, 1.0+0.020, 0.4+0.999] = [1.509,-0.216, 1.020, 1.399]
```

> ✅ Ahora cada token tiene información de **qué es** (embedding) y **dónde está** (posición).

---

## 🎯 Paso 2 — Self-Attention

### 2.1 — Definir las matrices de pesos

Se definen tres matrices aprendibles (aquí usamos valores pequeños de ejemplo):

```
W_Q = | 0.1  0.2 |    W_K = | 0.3  0.1 |    W_V = | 0.5  0.2 |
      | 0.3  0.1 |          | 0.2  0.4 |          | 0.1  0.4 |
      | 0.2  0.4 |          | 0.1  0.2 |          | 0.3  0.1 |
      | 0.1  0.3 |          | 0.4  0.3 |          | 0.2  0.3 |
```

_(Reducimos de 4 dimensiones a 2 para simplificar: d_k = 2)_

---

### 2.2 — Calcular Q, K, V

Multiplicamos X' por cada matriz de pesos. Mostramos el cálculo para **"gato"** `= [1.141, 1.540, 0.710, 1.099]`:

**Query de "gato":**

```
Q_gato = X'_gato · W_Q

q1 = 1.141×0.1 + 1.540×0.3 + 0.710×0.2 + 1.099×0.1 = 0.114+0.462+0.142+0.110 = 0.828
q2 = 1.141×0.2 + 1.540×0.1 + 0.710×0.4 + 1.099×0.3 = 0.228+0.154+0.284+0.330 = 0.996

Q_gato = [0.828, 0.996]
```

Haciendo lo mismo para todos los tokens, obtenemos:

```
Q = | 0.652  0.970 |   ← "El"
    | 0.828  0.996 |   ← "gato"
    | 0.643  0.880 |   ← "duerme"

K = | 0.987  0.982 |   ← "El"
    | 1.013  1.002 |   ← "gato"
    | 0.889  0.901 |   ← "duerme"

V = | 1.122  1.082 |   ← "El"
    | 0.987  1.054 |   ← "gato"
    | 1.096  0.976 |   ← "duerme"
```

---

### 2.3 — Calcular los Scores: QKᵀ

Multiplicamos Q por Kᵀ. Cada celda `(i,j)` responde: _¿cuánto debe atender el token i al token j?_

Ejemplo: score de **"gato"** mirando a **"El"**:

```
score(gato→El) = Q_gato · K_El = 0.828×0.987 + 0.996×0.982
               = 0.817 + 0.978 = 1.795
```

Matriz completa de scores:

```
QKᵀ =            "El"   "gato"  "duerme"
       "El"     | 1.598   1.634    1.456 |
       "gato"   | 1.795   1.839    1.609 |
       "duerme" | 1.499   1.535    1.364 |
```

---

### 2.4 — Escalar por √d_k

Con `d_k = 2`, dividimos todo entre `√2 ≈ 1.414`:

```
QKᵀ / √2 =       "El"   "gato"  "duerme"
       "El"     | 1.130   1.155    1.030 |
       "gato"   | 1.269   1.300    1.138 |
       "duerme" | 1.060   1.085    0.964 |
```

---

### 2.5 — Aplicar Softmax (fila por fila)

Para cada fila, aplicamos softmax: `softmax(zᵢ) = e^zᵢ / Σe^z`

**Fila de "gato"** = [1.269, 1.300, 1.138]:

```
e^1.269 = 3.557,  e^1.300 = 3.669,  e^1.138 = 3.121
Suma = 10.347

α(gato→El)      = 3.557 / 10.347 = 0.344
α(gato→gato)    = 3.669 / 10.347 = 0.355
α(gato→duerme)  = 3.121 / 10.347 = 0.302
                                    ───────
                                     1.001 ≈ 1.0 ✅
```

Matriz de atención completa:

```
Softmax(QKᵀ/√2) =    "El"   "gato"  "duerme"
          "El"     | 0.342   0.351    0.307 |
          "gato"   | 0.344   0.355    0.302 |
          "duerme" | 0.341   0.350    0.309 |
```

> 🔍 Los valores son similares porque nuestros embeddings de ejemplo son muy parecidos. En un modelo real entrenado, habría diferencias mucho más marcadas — por ejemplo "gato" atendería fuertemente a "duerme" (relación sujeto-verbo).


### Softmax explicado paso a paso

#### ¿Qué es Softmax y para qué sirve aquí?

Antes del Softmax, tienes una fila de scores como `[1.269, 1.300, 1.138]` — son números crudos que dicen "cuánto se relaciona gato con cada token". El problema es que son números arbitrarios y no se puede interpretar fácilmente su magnitud.

Softmax los **convierte en probabilidades** — números entre 0 y 1 que suman exactamente 1. Así puedes leerlos como _"porcentaje de atención"_.

#### La fórmula

```
softmax(zᵢ) = e^zᵢ / Σe^z
```

En español: _"toma el número, elévalo como potencia de e, y divídelo entre la suma de todos los demás también elevados a e"_.

**¿Por qué usar `e` (número de Euler ≈ 2.718)?** Porque amplifica las diferencias — los valores más altos crecen proporcionalmente más, haciendo que el ganador "destaque" más.

#### El cálculo para "gato" = [1.269, 1.300, 1.138]

**Paso 1 — Elevar cada número a e:**

```
e^1.269 = 2.718^1.269 = 3.557   ← atención hacia "El"
e^1.300 = 2.718^1.300 = 3.669   ← atención hacia "gato"
e^1.138 = 2.718^1.138 = 3.121   ← atención hacia "duerme"
```

**Paso 2 — Sumar todos:**

```
3.557 + 3.669 + 3.121 = 10.347
```

**Paso 3 — Dividir cada uno entre la suma:**

```
α(gato→El)      = 3.557 / 10.347 = 0.344  → 34.4% de atención
α(gato→gato)    = 3.669 / 10.347 = 0.355  → 35.5% de atención
α(gato→duerme)  = 3.121 / 10.347 = 0.302  → 30.2% de atención
                                             ──────
                                             1.001 ≈ 100% ✅
```

#### ¿Cómo leer la matriz completa?

```
                  "El"   "gato"  "duerme"
     "El"      | 0.342   0.351    0.307 |
     "gato"    | 0.344   0.355    0.302 |
     "duerme"  | 0.341   0.350    0.309 |
```

Cada **fila** es un token preguntando: _¿a quién le presto atención?_

Por ejemplo la fila de **"gato"** dice:

- Presto 34.4% de atención a "El"
- Presto 35.5% de atención a mí mismo
- Presto 30.2% de atención a "duerme"
#### ¿Por qué los valores son tan parecidos (~0.33 cada uno)?

Porque los embeddings del ejemplo son **inventados y muy similares entre sí**. En un modelo real entrenado con millones de textos, la fila de "gato" se vería más así:

```
                  "El"   "gato"  "duerme"
     "gato"    | 0.05    0.10     0.85   ← ¡"gato" atiende fuerte a "duerme"!
```

Porque el modelo habría aprendido que un **sustantivo** (gato) se relaciona fuertemente con su **verbo** (duerme). Eso es exactamente lo que el entrenamiento enseña.

#### Resumen intuitivo

Softmax es como **votar con presupuesto fijo**. Tienes 100 puntos de atención para repartir entre todos los tokens. Los tokens con score más alto se llevan más puntos, pero todos reciben algo y la suma siempre da 100%.

### 2.6 — Ponderar los Values

La salida de la atención es la suma ponderada de los Values:

```
Output = Softmax(QKᵀ/√2) · V
```

Para **"gato"** (pesos: 0.344, 0.355, 0.302):

```
out1 = 0.344×1.122 + 0.355×0.987 + 0.302×1.096
     = 0.386 + 0.351 + 0.331 = 1.068

out2 = 0.344×1.082 + 0.355×1.054 + 0.302×0.976
     = 0.372 + 0.374 + 0.295 = 1.041

Output_gato = [1.068, 1.041]
```

Salida completa del bloque de atención:

```
Output =  | 1.071  1.041 |   ← "El"     (enriquecido con contexto)
          | 1.068  1.041 |   ← "gato"   (enriquecido con contexto)
          | 1.069  1.040 |   ← "duerme" (enriquecido con contexto)
```

> ✅ Cada token ahora tiene una representación que **incorpora información de todos los otros tokens**.

---

## 🧠 Paso 3 — Multi-Head Attention

En lugar de calcular la atención una sola vez, se hacen **h cálculos en paralelo**, cada uno con sus propias matrices W_Q, W_K, W_V.

Con 2 cabezas (`h=2`) y nuestro ejemplo:

```
Cabeza 1 (relaciones sintácticas):
  → Usa W_Q1, W_K1, W_V1 (las del ejemplo anterior)
  → Output_head1 = | 1.071  1.041 |
                   | 1.068  1.041 |
                   | 1.069  1.040 |

Cabeza 2 (relaciones semánticas — con diferentes W):
  → Usa W_Q2, W_K2, W_V2
  → Output_head2 = | 0.921  1.102 |
                   | 0.943  1.088 |
                   | 0.912  1.095 |
                   (valores ilustrativos)
```

Se **concatenan** las salidas horizontalmente:

```
Concat(head1, head2) =
  | 1.071  1.041  0.921  1.102 |   ← "El"
  | 1.068  1.041  0.943  1.088 |   ← "gato"
  | 1.069  1.040  0.912  1.095 |   ← "duerme"
```

Y se proyecta con una matriz W_O (4×4) para obtener la salida final del bloque:

```
MultiHead Output = Concat · W_O
```

---

## ➕ Paso 4 — Conexión Residual + Layer Normalization

Después de cada subcapa (atención y feed-forward), se aplica:

```
Output = LayerNorm(x + SubLayer(x))
```

### ¿Por qué la conexión residual?

Se suma la entrada original con la salida de la subcapa. Esto permite que el gradiente fluya directamente hacia atrás sin desvanecerse.

```
x_gato original         = [1.141, 1.540, 0.710, 1.099]
Atención(x_gato)        = [1.068, 1.041, ...]   (proyectado de vuelta a dim 4)
Suma residual           = x_gato + Atención(x_gato)
                        = [2.209, 2.581, ...]
```

### Layer Normalization:

Normaliza cada vector para que tenga **media 0** y **desviación estándar 1**:

```
Para el vector z = [2.209, 2.581, 1.710, 2.199]:

  μ = (2.209+2.581+1.710+2.199)/4 = 2.175
  σ = desviación estándar ≈ 0.312

  z_norm = (z - μ) / σ = [0.109, 1.301, -1.490, 0.077]
```

> ✅ Esto estabiliza el entrenamiento y acelera la convergencia.

---

## ⚙️ Paso 5 — Feed-Forward Network

Después de la atención, cada token pasa por una red feed-forward **independiente** (los mismos pesos para todos los tokens, pero sin interacción entre ellos):

```
FFN(x) = ReLU(x · W₁ + b₁) · W₂ + b₂
```

Con dimensión interna de 4 (en la práctica es 4× la dimensión del modelo):

Para **"gato"** después de LayerNorm = `[0.109, 1.301, -1.490, 0.077]`:

```
Capa 1 (con ReLU):
  h = ReLU([0.109, 1.301, -1.490, 0.077] · W₁ + b₁)
  h = ReLU([0.823, 1.204, -0.312, 0.541])
  h =      [0.823, 1.204,  0.000, 0.541]   ← ReLU pone en 0 los negativos

Capa 2:
  output = h · W₂ + b₂
  output = [0.714, 1.052, 0.831, 0.623]    (resultado final del bloque)
```

> Se vuelve a aplicar **conexión residual + LayerNorm** después del FFN.

---

## 🔁 El Bloque Completo

Todo lo anterior (pasos 2 al 5) forma **un bloque Transformer**. Este bloque se apila N veces (el paper original usa N=6):

```
Input X'
   ↓
┌─────────────────────────────────┐
│         Bloque Transformer      │
│                                 │
│  Multi-Head Self-Attention      │
│         ↓ + x (residual)        │
│      Layer Norm                 │
│         ↓                       │
│   Feed-Forward Network          │
│         ↓ + x (residual)        │
│      Layer Norm                 │
└──────────────┬──────────────────┘
               ↓
         (repetir N veces)
               ↓
         Representación final
```

---

## 📊 Resumen Numérico del Flujo Completo

```
"El gato duerme"
       ↓
Embeddings:  dim=4, valores aprendidos
       ↓
+ Positional Encoding:  señal seno/coseno por posición
       ↓
Self-Attention:
  Q·Kᵀ → scores → /√2 → softmax → pesos de atención
  pesos · V → representaciones contextuales
       ↓
Multi-Head:  2 cabezas → concat → proyección W_O
       ↓
Residual + LayerNorm:  estabiliza y preserva gradiente
       ↓
Feed-Forward:  ReLU(xW₁+b₁)W₂+b₂  por cada token
       ↓
Residual + LayerNorm
       ↓
Representación final enriquecida de cada token
```

---

> 💡 **Lo que muestran los números:** Aunque con embeddings artificiales los pesos de atención resultan uniformes (~0.33 para cada token), en un modelo **entrenado con datos reales** los pesos serían muy distintos. "Gato" atendería fuertemente a "duerme" (el sujeto busca su verbo), y "duerme" atendería a "gato" (el verbo busca su sujeto). Eso es exactamente lo que el modelo **aprende** durante el entrenamiento.


### En clase

![[Pasted image 20260513101610.png]]
Nx: Significa que el bloque se repite "N" veces

### DE VITAL IMPORTANCIA:
![[Pasted image 20260513101624.png]]


![[AttentionIsAllYouNeed.pdf]]
