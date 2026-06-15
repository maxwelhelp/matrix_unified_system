# Matrix Pseudocode Decompiler: метод раскрытия нейросетевых весов в исполняемые матричные программы

Версия: v1  
Контекст: Qwen/Qwen2.5 attention-head experiments, June 2026  
Цель: зафиксировать метод так, чтобы его можно было повторять на разных головах, слоях, моделях и задачах.

---

## 0. Короткая идея

Обычный forward-pass модели даёт результат:

```python
prompt -> model.forward() -> logits -> next token
```

Но внутри это выглядит как миллиарды чисел: матрицы, softmax, gates, residual stream, MLP. Метод здесь переводит эти численные матрицы в **раскрытое матричное вычисление**, то есть в псевдокод вида:

```python
score_ij = (
    c_delta
  + q_affine_delta(x_i)
  + k_affine_delta(x_j)
  + content_bilinear_delta(x_i, x_j)
)

A = causal_softmax(score)

payload_j = C_vo @ x_j + b_vo

Y_i = sum_j A[i, j] * payload_j
```

Это не исходный Python-код модели. Это **эквивалентная матричная программа**, раскрывающая, какие части вычисления дают score, router, payload, write и итоговый contribution.

---

## 1. Что именно считается “псевдокодом в матричном пространстве”

Псевдокодом здесь называется не текстовый `if/else`, а исполняемая формула/программа, которая:

1. строится из реальных весов модели;
2. учитывает архитектуру: RMSNorm, RoPE, GQA, bias, causal mask, softmax;
3. раскладывает матрицы на смысловые члены;
4. может быть запущена вместо оригинального компонента;
5. проверяется численно по `score/A/Z/Y/logits/loss`.

Пример для одной attention head:

```python
# Original head
Q = q_proj(Xn)
K = k_proj(Xn)
V = v_proj(Xn)
Q, K = apply_rope(Q, K)
A = softmax(Q @ K.T / sqrt(d))
Y = (A @ V) @ Wo_head.T

# Matrix pseudocode form
score_ij = c_delta + q_affine_i + k_affine_j + content_pair_ij
A = softmax(score)
payload_j = C_vo @ Xn_j + b_vo
Y_i = sum_j A[i, j] * payload_j
```

---

## 2. Отличие от обычного circuit analysis

### 2.1 Что делает обычный circuit analysis

Обычные circuit methods обычно отвечают:

```python
какие компоненты и связи важны?
Head L2H1 -> Head L3H6 -> MLP L4 -> logit
```

Это карта причинного пути. Она полезна, но часто не раскрывает точную формулу внутри узла.

### 2.2 Что делает matrix-pseudocode decomposition

Метод здесь отвечает:

```python
какую вычислительную программу выполняет компонент?
какой член score выбирает токен?
какой payload записывается?
какой route сработал?
какой term можно выключить и получить предсказуемое изменение?
```

Например не просто:

```python
L2H1 важна
```

а:

```python
L2H1:
    score ≈ c_delta + q_affine + k_affine + content_bilinear
    c_delta ≈ 98% energy, но content/k_affine функционально важны
    payload = C_vo @ x + b_vo
    VO_bias обязателен для точного Y
```

---

## 3. Основные объекты метода

### 3.1 Raw weights

Для Qwen attention:

```python
Wq = layer.self_attn.q_proj.weight
bq = layer.self_attn.q_proj.bias
Wk = layer.self_attn.k_proj.weight
bk = layer.self_attn.k_proj.bias
Wv = layer.self_attn.v_proj.weight
bv = layer.self_attn.v_proj.bias
Wo = layer.self_attn.o_proj.weight
```

По голове:

```python
head_dim = hidden_size // num_attention_heads
kv_groups = num_heads // num_kv_heads
kv_idx = head_idx // kv_groups

Wq_h = Wq[head_idx*head_dim:(head_idx+1)*head_dim, :]
bq_h = bq[head_idx*head_dim:(head_idx+1)*head_dim]

Wk_h = Wk[kv_idx*head_dim:(kv_idx+1)*head_dim, :]
bk_h = bk[kv_idx*head_dim:(kv_idx+1)*head_dim]

Wv_h = Wv[kv_idx*head_dim:(kv_idx+1)*head_dim, :]
bv_h = bv[kv_idx*head_dim:(kv_idx+1)*head_dim]

Wo_h = Wo[:, head_idx*head_dim:(head_idx+1)*head_dim]
```

### 3.2 Homogeneous / affine coordinates

Обязательно учитывать bias. Для этого добавляем координату `1`:

```python
x_aug = [x, 1]
Wq_aug = [Wq | bq]
Wk_aug = [Wk | bk]
Wv_aug = [Wv | bv]
```

Форма:

```python
q = Wq_aug @ x_aug
k = Wk_aug @ x_aug
v = Wv_aug @ x_aug
```

Именно это исправило v1 → v2:

```python
без bias: qk_score_rel ≈ 0.99
с affine/bias: qk_score_rel ≈ 0.00018
```

### 3.3 RMSNorm dynamic diagonal

RMSNorm не является чёрным ящиком. Для токена `x`:

```python
rms = sqrt(mean(x**2) + eps)
Xn = gamma * x / rms
```

Приближённо как dynamic diagonal:

```python
D_t = diag(gamma / rms(x_t))
Xn_t = D_t @ raw_x_t
```

Для полного Jacobian есть ещё rank-1 correction, но для circuit-target часто хватает dynamic diagonal, если сравнение идёт с реальным `Xn`.

### 3.4 RoPE relative operator

Для Qwen/RoPE:

```python
q_rot_i = R_i @ q_i
k_rot_j = R_j @ k_j
q_rot_i.T @ k_rot_j = q_i.T @ (R_i.T @ R_j) @ k_j
```

Определяем:

```python
R_delta = R_i.T @ R_j
```

Тогда:

```python
M_qk_aug_delta = Wq_aug.T @ R_delta @ Wk_aug / sqrt(head_dim)
```

### 3.5 QK affine-bilinear score decomposition

`M_qk_aug_delta` имеет размер `(hidden+1) x (hidden+1)`.

Разбиваем:

```python
M_aug = [[B_delta, u_delta],
         [v_delta.T, c_delta]]
```

Тогда:

```python
score_ij = (
    Xn_i.T @ B_delta @ Xn_j
  + Xn_i.T @ u_delta
  + v_delta.T @ Xn_j
  + c_delta
)
```

Где:

- `B_delta`: content-bilinear pair interaction;
- `u_delta`: query-affine term;
- `v_delta`: key-affine term;
- `c_delta`: constant / relative-position / bias prior.

### 3.6 VO affine write decomposition

```python
C_vo_aug = Wo_head @ Wv_aug
```

Разбиваем:

```python
C_vo_aug = [C_vo_linear | b_vo]
```

Тогда:

```python
payload_j = C_vo_linear @ Xn_j + b_vo
Y_i = sum_j A[i, j] * payload_j
```

### 3.7 Full head operator for one prompt

Для фиксированного prompt и известного `A`:

```python
Y_i = sum_j A[i, j] * (C_vo_linear @ D_j @ raw_x_j + b_vo)
```

Block operator:

```python
Block(i, j) = A[i, j] * C_vo_linear @ D_j
```

Полная форма:

```python
Y_flat = W_head_prompt @ raw_X_flat + bias_from_attention
```

---

## 4. Уровни анализа

### Level 0: обычный forward-pass

```python
model(prompt) -> logits
```

Показывает результат, но не объясняет вычисления.

### Level 1: raw-weight scan

Смотрим на `Wq/Wk/Wv/Wo` отдельно.

Вопрос:

```python
сама матрица читается или серая?
```

Метрики:

```python
rank profile
block energy
diagonal/band energy
SVD spectrum
bias norm
```

Доверие: низкое/среднее. Если плохо — не значит, что смысла нет.

### Level 2: composite scan

Строим:

```python
QK = Wq.T @ Wk
VO = Wo @ Wv
```

Вопрос:

```python
какой circuit получается из пары весов?
```

Доверие выше, чем raw.

### Level 3: affine/RoPE/RMS-aware circuit targets

Строим:

```python
M_qk_aug_delta
C_vo_aug
D_rms_token
```

Это основной уровень для реального Qwen.

### Level 4: score-term decomposition

Разбиваем score:

```python
constant
query_affine
key_affine
content_bilinear
```

Тестируем term ablation:

```python
no_const
no_q
no_k
no_content
only_const
only_content
```

### Level 5: route-specific decomposition

Кластеризуем rows attention:

```python
route0 diffuse
route1 local/BOS
route2 forced self
```

И проверяем terms по route.

### Level 6: head bank / multihead decomposition

Собираем heads как operator bank:

```python
Y_bank = sum_h coef_h * HeadProgram_h
```

И обучаем gates:

```python
coef_head
gate_const
gate_q
gate_k
gate_content
gate_vo_linear
gate_vo_bias
```

### Level 7: causal control

Патчим term внутри модели:

```python
score_patch = score - k_affine_for_token
```

Смотрим:

```python
A change
Y change
logit change
KL
next token change
```

Если effect направленный и предсказуемый — это управление.

---

## 5. Проверки корректности

### 5.1 Exact reconstruction checks

Для головы:

```python
Qpre_from_affine_rel
Kpre_from_affine_rel
V_from_affine_Wv_rel
qk_score_aug_rel
A_rel_from_aug_scores
Y_from_Cvo_aug_rel
Y_block_raw_affine_rel
```

Хорошо:

```python
< 1e-3 примерно для fp16/eager experiments
```

### 5.2 Functional checks

Для QK:

```python
A_rel
KL
top1_match
Z_rel = ||A_hat V - A_true V|| / ||A_true V||
Y_rel = ||Z_hat Wo - Y_true|| / ||Y_true||
```

Для patch:

```python
logit_rel
KL_orig_to_patch
top1_match
loss_delta
```

### 5.3 Term ablation checks

Если term важен, при выключении должно расти:

```python
A_rel
Y_rel
logit_rel
KL
loss_delta
```

### 5.4 Basis validation

Плохой критерий:

```python
матрица выглядит красиво
```

Хороший критерий:

```python
basis держит score/A/Z/Y/logits
```

---

## 6. Типы базисов

### 6.1 Score-space SVD basis

Сжимаем сам score matrix/family.

Вопрос:

```python
есть ли компактный basis в score-space?
```

Из 10-head bank результат:

```python
score_svd rank16 почти exact по A/top1 для многих голов
```

Вывод:

```python
многие головы проще читать в score-space, чем через Q/K vector-basis
```

### 6.2 Learned QK Pq/Pk basis

Ищем:

```python
A_hat = softmax((Q @ Pq) @ (K @ Pk).T / sqrt(rank))
```

Цель:

```python
найти сжатые Q/K directions, которые держат A/Z/Y
```

### 6.3 Delta-family basis

Для families:

```python
c_delta
u_delta
v_delta
B_delta
```

Пример:

```python
c_delta ≈ a0*const + a1*linear + a2*quad + a3*decay2 + ...
```

Для vectors:

```python
stack u_delta over delta -> SVD rank family
stack v_delta over delta -> SVD rank family
```

### 6.4 Route-specific basis

Для каждого route:

```python
fit basis only on rows belonging to route
```

Вопрос:

```python
одна голова использует разные программы по режимам?
```

### 6.5 Head-bank basis

Сама модель даёт словарь:

```python
HeadProgram_1, HeadProgram_2, ..., HeadProgram_N
```

Разложение:

```python
Target ≈ sum_h a_h * HeadProgram_h
```

Проценты:

```python
percent_h = |a_h| * ||HeadProgram_h|| / sum_j |a_j| * ||HeadProgram_j||
```

Но нужно учитывать overlap/кластеры, потому что головы не ортогональны.

---

## 7. Основные эксперименты и варианты

### 7.1 Single-head affine circuit extraction

Цель: доказать, что одна голова раскладывается в exact affine matrix program.

Команда-шаблон:

```bash
python qwen_circuit_matrix_targets_v2_affine_basis.py \
  --base-script ./qwen_program_decompiler_v6_scorehybrid.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda \
  --dtype fp16 \
  --attn-implementation eager \
  --heads 2:1 \
  --prompt-suites all \
  --prompts-per-suite 8 \
  --same-text-repeats 2 \
  --max-length 192 \
  --max-delta 64 \
  --svd-ranks 4,8,16,32,64,128 \
  --basis-ranks 4,8,16,32,64 \
  --fit-learned-qk-basis \
  --learned-steps 120 \
  --out-dir ./qwen_circuit_targets_v2_L2H1_fast
```

Ожидаемые файлы:

```python
summary.json
basis_functional.csv
per_prompt_checks.csv
token_flow_examples.json
circuit_targets.pt
```

### 7.2 Nonstandard score-term experiment

Цель: не просто exact, а понять важность terms.

```bash
python qwen_nonstandard_circuit_experiment_v1.py \
  --circuit-script ./qwen_circuit_matrix_targets_v2_affine_basis.py \
  --base-script ./qwen_program_decompiler_v6_scorehybrid.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda \
  --dtype fp16 \
  --attn-implementation eager \
  --heads 2:1 \
  --prompt-suites all \
  --prompts-per-suite 8 \
  --max-length 192 \
  --routes 3 \
  --vo-ranks 4,8,16,32,64 \
  --family-ranks 1,2,4,8,16 \
  --out-dir ./qwen_nonstandard_circuit_v1_L2H1
```

Важные файлы:

```python
score_term_energy.csv
score_term_functional_overall.csv
score_term_functional_by_route.csv
vo_functional.csv
delta_family_mining.json
token_pair_score_decomposition.json
```

### 7.3 Raw-weight projection scan

Цель: смотреть серые веса по слоям без активаций.

```bash
python qwen_raw_weight_projection_lab_v1.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda \
  --dtype fp16 \
  --attn-implementation eager \
  --layers 0-5 \
  --heads all \
  --deltas 0,1,2,3,4,5,6,7,8,16,32,64 \
  --ranks 4,8,16,32,64 \
  --subspace-rank 16 \
  --transition-window 2 \
  --top-transitions 120 \
  --out-dir ./qwen_raw_weight_projection_L0_5
```

Смотрим:

```python
head_raw_projection_summary.csv
top_cross_layer_transitions.csv
top_same_role_subspace_overlaps.csv
```

### 7.4 Two-head bank probe

Цель: проверить, две головы дублируют друг друга или образуют путь.

```bash
python qwen_two_head_operator_bank_probe_v1.py \
  --base-script ./qwen_program_decompiler_v6_scorehybrid.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda \
  --dtype fp16 \
  --attn-implementation eager \
  --heads 2:1,3:6 \
  --prompt-suites all \
  --prompts-per-suite 8 \
  --max-length 192 \
  --subspace-rank 16 \
  --out-dir ./qwen_two_head_bank_L2H1_L3H6
```

Проверки:

```python
cosine(Y_A, Y_B)
A explained by B
B explained by A
writeA_to_B_Wq_read
writeA_to_B_Wv_read
writeB_to_A_Wq_read
```

### 7.5 Multihead differentiable bank

Цель: 10 голов как один дифференцируемый банк.

```bash
python qwen_multihead_differentiable_bank_v1_py312_fixed.py \
  --base-script ./qwen_program_decompiler_v6_scorehybrid.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda \
  --dtype fp16 \
  --attn-implementation eager \
  --heads 2:1,3:6,3:2,4:8,4:2,4:0,4:5,2:8,0:9,4:9 \
  --prompt-suites all \
  --prompts-per-suite 8 \
  --same-text-repeats 2 \
  --max-length 192 \
  --val-frac 0.25 \
  --bank-steps 250 \
  --bank-lr 0.03 \
  --bank-eval-every 25 \
  --bank-patience 8 \
  --sparse-lambda 0.002 \
  --subspace-rank 16 \
  --score-basis-ranks 4,8,16,32 \
  --fit-qk-bases \
  --qk-basis-ranks 4,8,16,32,64 \
  --qk-basis-steps 80 \
  --qk-basis-lr 0.003 \
  --out-dir ./qwen_multihead_bank_10_v1
```

Смотрим:

```python
per_head_exact_checks.csv
joint_bank_gates.csv
joint_term_ablation.csv
cross_space_overlaps.csv
qk_basis_functional.csv
score_basis_svd.csv
```

### 7.6 Term control probe

Цель: доказать управление, а не только описание.

```bash
python qwen_term_control_probe_v1.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda \
  --dtype fp16 \
  --attn-implementation eager \
  --head 3:6 \
  --prompt "Write a Python function that reverses a linked list." \
  --term k_affine \
  --target-key-pos 0 \
  --strength 1.0 \
  --out ./term_control_L3H6_k_affine_key0.json
```

Если `attention/logits` меняются предсказуемо — это control knob.

---

## 8. Интерпретация полученных типов голов

### 8.1 Positional-affine reader

Признаки:

```python
constant_delta% очень высокий
content_bilinear% низкий
```

Пример:

```python
L2H1:
    constant≈98.4%
    query≈1.5%
    content≈0.02%
```

Интерпретация:

```python
голова в основном задаётся relative-position/bias prior
```

### 8.2 Query-driven head

```python
query_affine% высокий
```

Пример:

```python
L0H9:
    query≈81.5%
```

Интерпретация:

```python
текущий токен сильно определяет, что он хочет читать
```

### 8.3 Key-affine selector

```python
key_affine% высокий
```

Пример:

```python
L3H6:
    key≈59.1%
```

Интерпретация:

```python
выбор токена сильно определяется key-side свойствами читаемых токенов
```

### 8.4 Content/key head

```python
content_bilinear% и key_affine% заметны
```

Примеры:

```python
L4H2:
    content≈34%, key≈25.5%
L4H0:
    content≈27.9%, key≈30%
```

Интерпретация:

```python
ближе к настоящей content-sensitive голове
```

### 8.5 VO-bias writer

```python
VO_bias% высокий
```

Интерпретация:

```python
прочитанный token пишет не только payload из X, но и сильный constant write
```

---

## 9. Межголовные переходы

Static transition candidate:

```python
write_space(head A) overlaps read_space(head B)
```

Например:

```python
L2H1 writes -> L3H6 reads through Wq
L3H6 writes -> L4H8 reads through Wq
```

Метрики:

```python
mean_sq_cos
max_sq_cos
coupling
```

Интерпретация:

```python
это кандидат на путь, но не доказательство
```

Доказательство требует patch:

```python
remove/replace Y_A
measure Q_B/A_B/Y_B/logits
```

---

## 10. Управление

### 10.1 Описание vs управление

Описание:

```python
k_affine важен для score
```

Управление:

```python
меняем/обнуляем k_affine для конкретного token
-> attention/logits меняются в предсказуемом направлении
```

### 10.2 Виды управления

```python
term ablation:
    remove const/q/k/content

token-targeted control:
    remove k_affine only for key token j

route-targeted control:
    remove term only for rows in route r

head-bank control:
    scale head coefficient coef_h

VO control:
    remove VO_bias or linear payload
```

### 10.3 Критерии успешного управления

```python
A_mass_target changes predictably
Y_delta matches predicted direction
logit_rel/KL changes
specific logits move up/down
replacement recovers behavior
```

---

## 11. Что считается провалом

### 11.1 Exact reconstruction bad

Если:

```python
V_from_affine_Wv_rel high
Qpre/Kpre high
```

Проверить:

```python
bias missing?
wrong head slice?
wrong GQA kv_idx?
RoPE sign/orientation?
RMSNorm input mismatch?
```

### 11.2 QK basis bad but score basis good

Вывод:

```python
читать надо в score-space, не через Q/K vector projections
```

### 11.3 Raw W bad but composite good

Вывод:

```python
смысл не в Wq/Wk отдельно, а в QK/VO composition
```

### 11.4 Static overlap high but causal patch low

Вывод:

```python
геометрический кандидат есть, но модель на данных его не использует
```

---

## 12. Расширение на MLP

MLP в Qwen примерно:

```python
gate = silu(W_gate x + b_gate)
up = W_up x + b_up
hidden = gate * up
out = W_down hidden
```

Локальный Jacobian:

```python
J_mlp(x) = W_down @ local_dynamic_matrix(x)
```

Псевдокод:

```python
mlp_gate_term = silu(gate_affine(x))
mlp_payload = up_affine(x)
mlp_hidden = mlp_gate_term * mlp_payload
mlp_write = W_down @ mlp_hidden
```

Разложения:

```python
gate directions
up payload directions
down write directions
local diagonal gate
product interactions gate*up
```

Проверки:

```python
J_rel
out_rel
logit patch
feature/token route dependence
```

---

## 13. Расширение на весь слой

Decoder layer:

```python
residual_in
  -> input RMSNorm
  -> attention all heads
  -> o_proj / residual add
  -> post-attn RMSNorm
  -> MLP gate/up/down
  -> residual add
```

Full layer pseudocode:

```python
attn_Y = sum_heads HeadProgram_h(X_norm)
h1 = residual_in + attn_Y
h1n = RMSNorm(h1)
mlp_Y = MLPProgram(h1n)
h2 = h1 + mlp_Y
```

---

## 14. Расширение на всю модель

Полная модель:

```python
embedding
for layer in layers:
    layer_pseudocode
final_rms
lm_head
logits
```

Для генерации:

```python
for step in generation:
    trace all layers/heads/MLP
    decompose logits
    sample/argmax next token
    append token
```

Выход:

```python
next token
why-trace:
    which heads moved logits
    which score terms selected tokens
    which MLP gates fired
    which residual directions reached lm_head
```

---

## 15. Что нужно для claim “мы объясняем поведение”

Минимум:

```python
1. exact reconstruction per component
2. term decomposition
3. functional ablation
4. replacement patch
5. control intervention
6. multi-token/multi-prompt validation
7. reject false positives
```

Сильный claim:

```python
for this model subset / heads / layer / prompts:
    matrix pseudocode predicts and controls behavior
```

Нельзя пока говорить:

```python
we fully solved black-box behavior for all inputs
```

Корректно:

```python
we provide an architecture-aware method for translating weights/circuits into executable matrix pseudocode with causal validation
```

---

## 16. Минимальные code snippets

### 16.1 Build affine matrix

```python
def affine_weight(W, b):
    # W: [out, in], b: [out]
    if b is None:
        b = torch.zeros(W.shape[0], device=W.device, dtype=W.dtype)
    return torch.cat([W, b[:, None]], dim=1)
```

### 16.2 Split QK augmented matrix into terms

```python
def split_aug_matrix(M_aug):
    B = M_aug[:-1, :-1]
    u = M_aug[:-1, -1]
    v = M_aug[-1, :-1]
    c = M_aug[-1, -1]
    return B, u, v, c
```

### 16.3 Score terms

```python
def score_terms(X, B, u, v, c):
    # X: [T, H]
    content = X @ B @ X.T
    q_aff = X @ u
    k_aff = X @ v
    score = content + q_aff[:, None] + k_aff[None, :] + c
    return {
        "content": content,
        "q_affine": q_aff[:, None].expand_as(content),
        "k_affine": k_aff[None, :].expand_as(content),
        "constant": torch.full_like(content, float(c)),
        "score": score,
    }
```

### 16.4 Causal mask softmax

```python
def causal_softmax(score):
    T = score.shape[0]
    mask = torch.triu(torch.ones(T, T, device=score.device, dtype=torch.bool), diagonal=1)
    return torch.softmax(score.masked_fill(mask, -1e9), dim=-1)
```

### 16.5 VO payload

```python
def vo_payload(X, C_vo_aug):
    C = C_vo_aug[:, :-1]
    b = C_vo_aug[:, -1]
    return X @ C.T + b
```

### 16.6 Head pseudocode execution

```python
def run_head_pseudocode(X, M_aug_by_delta, C_vo_aug):
    T, H = X.shape
    score = torch.zeros(T, T, device=X.device)
    for i in range(T):
        for j in range(i + 1):
            d = i - j
            B, u, v, c = split_aug_matrix(M_aug_by_delta[d])
            score[i, j] = X[i] @ B @ X[j] + X[i] @ u + v @ X[j] + c
    A = causal_softmax(score)
    P = vo_payload(X, C_vo_aug)
    Y = A @ P
    return Y, A, score
```

Vectorize this for speed in production.

### 16.7 Term ablation

```python
def ablate_score(score_terms, remove):
    score = score_terms["score"].clone()
    if remove == "k_affine":
        score -= score_terms["k_affine"]
    elif remove == "q_affine":
        score -= score_terms["q_affine"]
    elif remove == "content":
        score -= score_terms["content"]
    elif remove == "constant":
        score -= score_terms["constant"]
    return score
```

### 16.8 Head-bank fit

```python
# Y_heads: [N_heads, N_rows, H]
# Y_target: [N_rows, H]
coef = torch.nn.Parameter(torch.ones(N_heads))
opt = torch.optim.Adam([coef], lr=0.03)

for step in range(steps):
    Y_hat = (coef[:, None, None] * Y_heads).sum(dim=0)
    loss = ((Y_hat - Y_target) ** 2).mean() + sparse_lambda * coef.abs().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
```

---

## 17. Practical checklist

Для новой модели/головы:

```python
[ ] проверить slices heads/GQA
[ ] проверить bias exists
[ ] проверить RoPE orientation
[ ] собрать X/Q/K/V/A/Y через реальный forward
[ ] собрать affine QK/VO
[ ] exact check Q/K/V/score/A/Y
[ ] term decomposition
[ ] term ablation
[ ] basis search score-space and QK-space
[ ] route clustering
[ ] route-specific term ablation
[ ] raw-weight scan across layers
[ ] transition overlaps
[ ] multihead bank
[ ] causal term control patch
[ ] document accepted/rejected heads
```

---

## 18. Ключевые выводы из текущих экспериментов

### L2H1 single-head

```python
qk_score_aug_rel ≈ 0.00018
A_rel ≈ 0.0013
Y_block_raw_affine_rel ≈ 0.000316
```

L2H1 type:

```python
RMS-scaled affine positional reader + VO payload writer
```

### Nonstandard term experiment

```python
content_bilinear small by energy but functionally important
key_affine critical
VO_bias important
```

### Raw scan L0-L5

```python
84 heads scanned
constant/query/key/content profiles differ by layer
VO_bias grows in later early/mid layers
cross-layer candidates found: L2H1 -> L3H6 -> L4H8
```

### Two-head L2H1 + L3H6

```python
not duplicates
L2H1 writes, L3H6 can read through Wq/Wv
```

### 10-head differentiable bank

```python
per-head exact Y_rel_mean ≈ 0.0016
joint bank val_rel ≈ 0.005
no_vo_bias / no_k / no_content strongly break bank
score-space rank16 basis very strong
```

---

## 19. Самая короткая формула метода

```python
weights + architecture
    -> circuit targets
    -> affine/RoPE/RMS matrix program
    -> term decomposition
    -> basis search
    -> residual mining
    -> executable pseudocode
    -> functional/causal validation
```

---

## 20. Как формулировать научно

Не говорить:

```python
we fully solved black-box neural networks
```

Говорить:

```python
We propose an architecture-aware matrix pseudocode decompilation method.
It translates selected neural network components from raw weights and code-paths into executable matrix programs, decomposes these programs into operator terms, and validates them by reconstruction, ablation, replacement, and causal control.
```

По-русски:

```python
Мы предлагаем метод архитектурно-осознанной декомпиляции нейросетевых компонентов в матричный псевдокод: реальные веса и код модели переводятся в исполняемые матричные программы, которые раскладываются на операторные члены и проверяются реконструкцией, абляцией, заменой и управляемыми вмешательствами.
```
