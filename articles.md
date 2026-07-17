# Статьи и материалы по структурным солверам оптимального управления

Снимок литературы и локальных находок, собранных в ходе сессии 16–17 июля
2026 года. Это не исчерпывающий обзор всей области, а тематический указатель
именно по обсуждавшимся вопросам:

- FATROP, MOTO/Hippo и отличие от IPOPT;
- Riccati-рекурсии для OCP/MOCP;
- глобальные статические переменные как оптимизируемые параметры;
- прямая коллокация;
- неявная динамика и DAE;
- ACADO/acados и близкие структурные солверы;
- возможное обобщение FATROP.

Обозначения статуса:

- **journal/conference** — опубликованная рецензируемая работа;
- **preprint** — препринт, выводы следует воспринимать с соответствующей
  осторожностью;
- **thesis** — диссертация или выпускная работа;
- **repository material** — код, ветка или внутренний документ, а не
  опубликованная статья.

## Краткий итог

1. **Riccati-рекурсия не ограничена multiple shooting как математический
   приём.** Она требует цепочной или древовидной структуры линейно-квадратичной
   подзадачи. Прямая коллокация обычно сначала локально сворачивается или
   переформулируется так, чтобы восстановить такую структуру.

2. **Для коллокации наиболее прямой маршрут к FATROP — lifted collocation.**
   Внутренние collocation variables и defect equations обрабатываются внутри
   интервала, а наружу передаётся multiple-shooting-подобный stage block.

3. **Глобальный вектор параметров создаёт bordered/arrow KKT-систему.** При
   небольшом числе параметров её можно решать как Riccati-факторизацию цепочной
   части плюс плотный Schur complement по глобальным переменным.

4. **Само по себе “добавить глобальные параметры в Riccati” уже недостаточно
   для заявления научной новизны.** К июлю 2026 года Sousa-Pinto–Orban уже
   включают одинарный global cross-stage vector в полный regularized
   primal-dual IPM; Martinez et al. дают параметризованную
   equality-constrained recursion, а PIQP — block-tridiagonal-arrow backend с
   global variables.

5. **Локальные linking variables/constraints в arrowhead KKT также уже prior
   art.** Rehfeldt et al. (2022) выводят разреженный Schur complement для
   связей соседних блоков, а Kempke–Rehfeldt–Koch (2026) строят точную
   иерархическую Schur-факторизацию и прямо отмечают расширение на несколько
   соседних блоков.

6. Потенциально новая и практически ценная работа должна объединять более
   сильный набор свойств: нелинейный primal-dual IPM/SQP, точный
   неопределённый Hessian, stage constraints, произвольные ограничения на
   глобальные параметры, неявную прямую коллокацию и линейную по горизонту
   сложность при фиксированной размерности глобального блока.

---

## 1. FATROP, базовая Riccati-рекурсия и IPOPT

### 1.1. FATROP

**L. Vanroye, A. Sathya, J. De Schutter, W. Decré.  
“FATROP: A Fast Constrained Optimal Control Problem Solver for Robot
Trajectory Optimization and Control.” IROS, 2023, pp. 10036–10043.**
**[conference]**

- [DOI](https://doi.org/10.1109/IROS55552.2023.10342336)
- [arXiv](https://arxiv.org/abs/2303.16746)
- Даёт: primal-dual interior-point solver, использующий структуру
  дискретного OCP и Riccati-подобное решение KKT-системы.
- Важно для сравнения с IPOPT: FATROP не отдаёт всю KKT-систему generic sparse
  factorization, а использует известную временную структуру.
- Граница: исходная публичная постановка — explicit discrete dynamics и
  stagewise constraints; глобальные оптимизируемые параметры и общая прямая
  коллокация не являются штатными сущностями интерфейса.

### 1.2. Equality-constrained Riccati

**L. Vanroye, J. De Schutter, W. Decré.  
“A Generalization of the Riccati Recursion for Equality-Constrained Linear
Quadratic Optimal Control.” Optimal Control Applications and Methods,
45(1):436–454, 2024.** **[journal]**

- [DOI](https://doi.org/10.1002/oca.3064)
- [arXiv](https://arxiv.org/abs/2302.14836)
- Даёт: generalized Riccati recursion для LQ-задач со stagewise equality
  constraints и аккуратной обработкой ранга.
- Это основная математическая база OCP linear solver в FATROP.
- Граница: глобальный одинарный decision block, связанный со всеми стадиями,
  напрямую в постановку статьи не входит.

### 1.3. Ранний IPM для MPC

**C. V. Rao, S. J. Wright, J. B. Rawlings.  
“Application of Interior-Point Methods to Model Predictive Control.”
Journal of Optimization Theory and Applications, 99(3):723–757, 1998.**
**[journal]**

- [DOI](https://doi.org/10.1023/A:1021711402723)
- Даёт: классическую связь primal-dual IPM, структуры MPC и решения
  последовательности structured Newton systems.
- Полезна как историческая база для сравнения Riccati-IPM с generic NLP/IPOPT.

### 1.4. Высокопроизводительная реализация Riccati

**G. Frison, J. B. Jørgensen.  
“Efficient Implementation of the Riccati Recursion for Solving
Linear-Quadratic Control Problems.” CCA, 2013, pp. 1117–1122.**
**[conference]**

- [DOI](https://doi.org/10.1109/CCA.2013.6662901)
- [DTU record](https://orbit.dtu.dk/en/publications/efficient-implementation-of-the-riccati-recursion-for-solving-lin/)
- Даёт: практические детали dense kernels, компоновки данных и эффективной
  реализации Riccati.

**G. Frison, L. E. Sokoler, J. B. Jørgensen.  
“A Family of High-Performance Solvers for Linear Model Predictive Control.”
IFAC Proceedings Volumes, 47(3):3074–3079, 2014.** **[conference]**

- [DOI](https://doi.org/10.3182/20140824-6-ZA-1003.01302)
- [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1474667016420793)
- Даёт: семейство реализаций для разных способов эксплуатации структуры и
  аппаратных платформ.

### 1.5. HPIPM и tree-structured QP

**G. Frison, M. Diehl.  
“HPIPM: a High-Performance Quadratic Programming Framework for Model
Predictive Control.” IFAC-PapersOnLine, 53(2):6563–6569, 2020.**
**[conference]**

- [DOI](https://doi.org/10.1016/j.ifacol.2020.12.073)
- [arXiv](https://arxiv.org/abs/2003.02547)
- Даёт: высокопроизводительный QP backend для OCP QP, partial condensing и
  dense QP, используемый, в частности, экосистемой acados.
- Граница: это QP solver, а не самостоятельный общий nonlinear NLP solver.

**G. Frison, D. Kouzoupis, M. Diehl, J. B. Jørgensen.  
“A High-Performance Riccati Based Solver for Tree-Structured Quadratic
Programs.” IFAC-PapersOnLine, 50(1):14399–14405, 2017.**
**[conference]**

- [DOI](https://doi.org/10.1016/j.ifacol.2017.08.2027)
- Даёт: расширение временной цепочки до дерева сценариев.
- Полезно как пример того, что Riccati — это метод эксплуатации графа
  разреженности, а не синоним исключительно multiple shooting.

### 1.6. Регуляризованная Riccati-рекурсия

**J. Sousa-Pinto, D. Orban.  
“Dual-Regularized Riccati Recursions for Interior-Point Optimal Control.”
arXiv:2509.16370, версия 6 от 15 июня 2026.** **[preprint]**

- [arXiv](https://arxiv.org/abs/2509.16370)
- Даёт: последовательную `O(N)` и параллельную `O(log N)` Riccati-рекурсии для
  dual-regularized LQR-подзадач primal-dual IPM, stagewise equalities и
  inequalities без требований полного ранга к Jacobian constraints.
- Раздел о **cross-stage variables** вводит один оптимизируемый вектор `theta`,
  существующий в единственном экземпляре и входящий в callbacks многих стадий.
  После `p+1` Riccati solves решается плотный Schur complement по `theta` с
  добавочной стоимостью `O(p^3)`.
- Особенно полезна для вопроса о робастной inertia correction при точном
  неопределённом Hessian.
- Граница: рассматривается discrete-time multiple shooting и один общий
  глобальный scope; иерархия phase/interface/segment scopes и прямая
  коллокация не являются предметом статьи.

### 1.7. IPOPT как generic sparse NLP baseline

**A. Wächter, L. T. Biegler.  
“On the Implementation of an Interior-Point Filter Line-Search Algorithm for
Large-Scale Nonlinear Programming.” Mathematical Programming, 106:25–57,
2006.** **[journal]**

- [DOI](https://doi.org/10.1007/s10107-004-0559-y)
- [IBM Research](https://research.ibm.com/publications/on-the-implementation-of-an-interior-point-filter-line-search-algorithm-for-large-scale-nonlinear-programming)
- Даёт: основной алгоритмический фундамент IPOPT — filter line search,
  primal-dual barrier method, restoration phase и sparse symmetric
  indefinite linear algebra.
- Сильная сторона IPOPT: почти произвольная гладкая NLP-постановка, включая
  прямую коллокацию, DAE и статические decision variables.
- Цена универсальности: temporal/OCP structure обычно видна лишь как общая
  sparse matrix и не превращается автоматически в специализированную
  Riccati-рекурсию.

### 1.8. CasADi как слой моделирования

**J. A. E. Andersson, J. Gillis, G. Horn, J. B. Rawlings, M. Diehl.  
“CasADi: a Software Framework for Nonlinear Optimization and Optimal
Control.” Mathematical Programming Computation, 11(1):1–36, 2019.**
**[journal]**

- [DOI](https://doi.org/10.1007/s12532-018-0139-4)
- Даёт: symbolic/algorithmic differentiation, генерацию функций и общий слой
  между OCP-моделированием и NLP/QP solvers.
- Важно не смешивать CasADi с linear solver: CasADi может сформировать нужную
  задачу, но эксплуатация MOCP-структуры определяется backend solver.

---

## 2. MOTO/Hippo, DDP и stagewise SQP

### 2.1. Hippo — ближайшая найденная публикация к архитектуре MOTO

**H. Zhao, L. Righetti, M. Khadiv.  
“Hippo: High-performance Interior-Point and Projection-based Solver for
Generic Constrained Trajectory Optimization.” arXiv:2603.00871, 2026.**
**[preprint]**

- [arXiv](https://arxiv.org/abs/2603.00871)
- Даёт: stagewise trajectory optimizer, inequality constraints через IPM,
  hard equalities через projection/nullspace либо IPM, сравнения с FATROP,
  acados и Aligator.
- По архитектуре и авторству это наиболее близкая найденная работа к
  локальному `moto`.
- **Оговорка:** в README репозитория MOTO нет явной ссылки, утверждающей, что
  MOTO и Hippo — один и тот же solver или что MOTO является его официальным
  кодом. Поэтому связь следует считать очень вероятной, но не документированной
  самим репозиторием.
- Граница: постановка основана на direct multiple shooting; один глобальный
  оптимизируемый parameter vector и direct collocation не продемонстрированы.

### 2.2. Crocoddyl

**C. Mastalli et al.  
“Crocoddyl: An Efficient and Versatile Framework for Multi-Contact Optimal
Control.” ICRA, 2020, pp. 2536–2542.** **[conference]**

- [DOI](https://doi.org/10.1109/ICRA40945.2020.9196673)
- [arXiv](https://arxiv.org/abs/1909.04947)
- Даёт: DDP/FDDP-подход к робототехнической trajectory optimization и
  эффективную эксплуатацию динамической структуры.
- Граница: исходный DDP особенно естественен для explicit dynamics и не
  является полной заменой generic constrained NLP.

### 2.3. Inverse-dynamics MPC через nullspace

**C. Mastalli, S. P. Chhatoi, T. Corbères, S. Tonneau, S. Vijayakumar.  
“Inverse-Dynamics MPC via Nullspace Resolution.” IEEE Transactions on
Robotics, 39(4):3222–3241, 2023.** **[journal]**

- [DOI](https://doi.org/10.1109/TRO.2023.3262186)
- [arXiv](https://arxiv.org/abs/2209.05375)
- Даёт: использование nullspace/projection для equality constraints в
  inverse-dynamics formulation и structured optimal control.
- Полезна для понимания projection path в MOTO/Hippo.

### 2.4. ProxDDP

**W. Jallet, A. Bambade, E. Arlaud, S. El-Kazdadi, N. Mansard,
J. Carpentier.  
“PROXDDP: Proximal Constrained Trajectory Optimization.” IEEE Transactions
on Robotics, 41, 2025.** **[journal]**

- [DOI](https://doi.org/10.1109/TRO.2025.3554437)
- [Project page](https://simple-robotics.github.io/publications/proxddp-tro-2025/)
- Даёт: proximal constrained DDP и современную точку сравнения для
  constrained trajectory optimization.

### 2.5. Structure-exploiting SQP для MPC

**A. Jordana, S. Kleff, A. Meduri, J. Carpentier, N. Mansard, L. Righetti.  
“Structure-Exploiting Sequential Quadratic Programming for Model-Predictive
Control.” IEEE Transactions on Robotics, 41:4960–4974, 2025.**
**[journal]**

- [DOI](https://doi.org/10.1109/TRO.2025.3595674)
- [Project/HAL page](https://gepettoweb.laas.fr/articles/jordana_kleff_meduri__sqp_tro_2025.html)
- Даёт: SQP, сохраняющий структуру OCP, и практический контекст для
  real-time robotics MPC.

### 2.6. Horizon

**F. Ruscelli, A. Laurenzi, N. G. Tsagarakis, E. Mingo Hoffman.  
“Horizon: A Trajectory Optimization Framework for Robotic Systems.”
Frontiers in Robotics and AI, 9, 2022.** **[journal/software article]**

- [DOI](https://doi.org/10.3389/frobt.2022.899025)
- [Open-access article](https://pmc.ncbi.nlm.nih.gov/articles/PMC9326239/)
- Даёт: единый framework с direct collocation, GN-SQP/OSQP и iLQR-like
  подходами; полезен как практическое сравнение транскрипций.
- Моделирующий API отдельно поддерживает `single variable`: один decision
  variable, используемый на нескольких узлах. Поэтому требуемый глобальный
  статический параметр выразить на уровне Horizon NLP можно.
- Граница: из статьи не следует, что custom GN-SQP/iLQR backend использует
  специальную bordered Riccati factorization для такого `single variable`.
  Generic NLP backend сохранит функциональность, но не обязательно даст
  искомую структурную скорость.

---

## 3. Глобальные статические decision variables

Под “глобальным параметром” здесь понимается не фиксированное число, переданное
модели, а оптимизируемая переменная

$$
\theta \in \mathbb{R}^{n_\theta},
$$

которая входит в функции многих или всех стадий, но существует в NLP в одном
экземпляре.

### 3.1. Параметризованная equality-constrained Riccati-рекурсия

**S. Martinez, S. Tonneau, C. Mastalli.  
“System Identification under Constraints and Disturbance: A Bayesian
Estimation Approach.” arXiv:2602.16358, 2026.** **[preprint]**

- [arXiv](https://arxiv.org/abs/2602.16358)
- Даёт: совместную оптимизацию state trajectory и глобальных физических
  параметров; equality/inequality priors на параметры; параметризованную
  equality-constrained Riccati-рекурсию с линейной сложностью по горизонту.
- Это наиболее прямой найденный prior art против слишком широкого утверждения
  “Riccati с глобальными параметрами ещё не существует”.
- Граница: специализированная Bayesian system-identification постановка,
  multiple shooting и equality-constrained stage structure; это ещё не
  универсальный nonlinear IPM для arbitrary collocation NLP.

### 3.2. Глобальные switching times

**S. Katayama, T. Ohtsuka.  
“Riccati Recursion for Optimal Control Problems of Nonlinear Switched
Systems.” arXiv:2102.02065, 2021.** **[preprint]**

- [arXiv](https://arxiv.org/abs/2102.02065)
- Даёт: Riccati-type structured Newton method, где switching instants являются
  one-copy оптимизируемыми величинами, встроенными в специальные граничные
  Riccati-шаги; заявлена `O(N)` сложность.
- Это близкий prior-art для интерфейсной переменной, действующей около границы
  режимов, а не только для полностью глобального border.

**S. Katayama, T. Ohtsuka.  
“Structure-Exploiting Newton-Type Method for Optimal Control of Switched
Systems.” International Journal of Control, 97(8):1717–1733, 2024.**
**[journal]**

- [DOI](https://doi.org/10.1080/00207179.2023.2227285)
- [arXiv](https://arxiv.org/abs/2112.07232)
- Даёт: более полную Newton-type формулировку и реализацию для switched OCP.

### 3.3. Block-tridiagonal-arrow QP

**R. Schwan, D. Kuhn, C. N. Jones.  
“Exploiting Multistage Optimization Structure in Proximal Solvers.” CDC,
2025, pp. 4677–4683.** **[conference]**

- [DOI](https://doi.org/10.1109/CDC57313.2025.11313030)
- [arXiv](https://arxiv.org/abs/2503.12664)
- Даёт: multistage QP с полной связью соседних стадий и global decision
  variables в cost, equality и inequality constraints; специализированную
  block-tridiagonal-arrow Cholesky factorization в PIQP.
- Это практически точная линейно-алгебраическая структура, возникающая при
  добавлении одного глобального параметрического блока к OCP.
- Граница: backend решает convex QP/proximal-IPM подзадачу, а не весь
  нелинейный OCP с exact indefinite Hessian.

**F. Song, R. Schwan, Y. Chen, C. N. Jones.  
“Parallel Structure-Exploiting Multistage Optimization in PIQP.” ICRA
Frontiers of Optimization for Robotics Workshop, 2026.**
**[workshop paper]**

- [OpenReview](https://openreview.net/forum?id=p8mJ3I0PSW)
- [PDF](https://openreview.net/pdf?id=p8mJ3I0PSW)
- Даёт: параллельную реализацию той же block-tridiagonal-arrow идеи.

### 3.4. Arrowhead IPM с локальными linking blocks

**D. Rehfeldt, H. Hobbie, D. Schönheit, T. Koch, D. Möst, A. Gleixner.
“A Massively Parallel Interior-Point Solver for LPs with Generalized
Arrowhead Structure, and Applications to Energy System Models.” European
Journal of Operational Research, 296(1):60–71, 2022.** **[journal]**

- [DOI](https://doi.org/10.1016/j.ejor.2021.06.063)
- Даёт: primal-dual IPM и Schur decomposition для doubly-bordered
  block-diagonal LP; большинство linking constraints связывает два соседних
  блока, что после перестановки даёт doubly-bordered block-tridiagonal Schur
  complement и линейный рост числа ненулей.
- Статья отдельно замечает, что аналогичная структура возникает для linking
  variables, связывающих два блока. Это прямой generic precedent для
  interface-scoped переменных.

**N.-C. Kempke, D. Rehfeldt, T. Koch.
“A Massively Parallel Interior-Point Method for Arrowhead Linear Programs
with Local Linking Structure.” SIAM Journal on Scientific Computing,
48(3):B360–B385, 2026.** **[journal]**

- [DOI](https://doi.org/10.1137/24M1716288)
- [arXiv](https://arxiv.org/abs/2412.07731)
- Даёт: exact hierarchical Schur-complement factorization в PIPS-IPM++ для
  linking constraints, действующих на несколько последовательных блоков.
- Авторы прямо отмечают, что multiple-adjacent-block constraints и локальность
  linking variables обрабатываются аналогично и что схема естественно
  переносится на structured nonlinear programs.
- Важная граница для нашей гипотезы: локальность border, иерархический Schur и
  линейное масштабирование сами по себе уже не новы. В PIPS локальные linking
  variables при этом не реализованы, а произвольная система перекрывающихся
  interval-support blocks, OCP Riccati leaves и implicit collocation не
  являются предметом экспериментов.

### 3.5. Interval-scoped Riccati — текущая исследовательская гипотеза

**“Interval-Scoped Riccati Factorization.”** **[repository material / research
hypothesis]**

- [Локальная записка](fatrop_implicit/research/INTERVAL_SCOPED_RICCATI.md)
- Идея: каждой one-copy переменной назначается непрерывный интервал фаз —
  phase-local `[f,f]`, interface `[f,f+1]`, segment `[a,b]` или global
  `[0,F-1]`. После независимого исключения trajectory-блоков reduced graph
  является interval graph.
- Порядок по возрастающей правой границе интервала является perfect
  elimination ordering, поэтому block factorization не создаёт структурного
  fill. Сложность ограничивается максимальной одновременно активной
  размерностью `omega`, а не суммой всех scoped variables.
- Уже реализованы reduced-KKT, complete implicit-OCP и полный standalone
  nonlinear exact-Hessian primal-dual Radau predictor-corrector с
  двусторонними path inequalities; проверены dense-reference и matched
  IPOPT+MA57 sweeps на 1/2/4/8 фазах и разных `(nx,nu,p)`.
- **Оговорка:** perfect elimination interval graph — известный факт. Возможная
  новизна после учёта PIPS ещё уже: явная реализованная поддержка произвольных
  перекрывающихся one-copy interval variables, interval-specific PEO/width
  bound и их интеграция с implicit/rank-aware Riccati leaves, nonlinear
  direct collocation и slack/complementarity reduction. Не завершены
  интеграция в production `IpAlgorithm`, restoration/inertia/globalization и
  независимая экспертиза новизны.

### 3.6. Общий nonlinear block-structured NLP с общими переменными

**J. Kang, Y. Cao, D. P. Word, C. D. Laird.  
“An Interior-Point Method for Efficient Solution of Block-Structured NLP
Problems Using an Implicit Schur-Complement Decomposition.” Computers &
Chemical Engineering, 71:563–573, 2014.** **[journal]**

- [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0098135414002798)
- Даёт: nonlinear IPM для block-structured NLP, где общий блок переменных
  связывает локальные подсистемы; implicit Schur-complement decomposition.
- Очень близка к общему algebraic pattern “локальная временная структура плюс
  глобальные параметры”, хотя внутренние блоки решаются не обязательно
  Riccati-рекурсией.

### 3.7. Базовая статья PIQP

**R. Schwan, Y. Jiang, D. Kuhn, C. N. Jones.  
“PIQP: A Proximal Interior-Point Quadratic Programming Solver.” CDC, 2023,
pp. 1088–1093.** **[conference]**

- [DOI](https://doi.org/10.1109/CDC49753.2023.10383915)
- [arXiv](https://arxiv.org/abs/2304.00290)
- Даёт: proximal primal-dual IPM, на котором построен более поздний
  multistage-arrow backend.

### 3.8. Линейная алгебра bordered Riccati

После линеаризации глобальные параметры дают KKT-систему вида

$$
\begin{bmatrix}
K & B \\
B^\top & C
\end{bmatrix}
\begin{bmatrix}
\Delta z \\
\Delta \theta
\end{bmatrix}
=-
\begin{bmatrix}
r_z \\
r_\theta
\end{bmatrix},
$$

где `K` — цепочная OCP-часть, а `B` связывает глобальный вектор со стадиями.
Её естественное решение:

1. факторизовать `K` обычной Riccati-рекурсией;
2. решить системы с несколькими правыми частями для `K^{-1}B` и
   `K^{-1}r_z`;
3. собрать плотный Schur complement
   `S = C - B^T K^{-1} B`;
4. решить систему размера `n_theta`;
5. выполнить обратную подстановку по цепочке.

При фиксированном `n_theta` сложность остаётся линейной по числу стадий:
приблизительно одна факторизация и `n_theta + 1` проходов с правой частью,
плюс `O(n_theta^3)` для плотного border. Это не отдельная новая идея, а
стандартный bordered/Schur-complement pattern; новизна может быть в его полном
обобщении, устойчивости и интеграции с конкретным nonlinear solver.

---

## 4. Прямая коллокация и восстановление цепочной структуры

### 4.1. Классическая коллокация

**L. T. Biegler.  
“Solution of Dynamic Optimization Problems by Successive Quadratic
Programming and Orthogonal Collocation.” Computers & Chemical Engineering,
8(3–4):243–247, 1984.** **[journal]**

- [DOI](https://doi.org/10.1016/0098-1354(84)87012-X)
- Даёт: одну из ранних SQP + orthogonal collocation постановок dynamic
  optimization.

**M. Kelly.  
“An Introduction to Trajectory Optimization: How to Do Your Own Direct
Collocation.” SIAM Review, 59(4):849–904, 2017.** **[journal/tutorial]**

- [DOI](https://doi.org/10.1137/16M1062569)
- Даёт: ясное введение в direct transcription, Hermite–Simpson и практическую
  структуру collocation NLP.

### 4.2. Lifted collocation — главный мост к Riccati

**R. Quirynen, S. Gros, B. Houska, M. Diehl.  
“Lifted Collocation Integrators for Direct Optimal Control in ACADO
Toolkit.” Mathematical Programming Computation, 9(4):527–571, 2017.**
**[journal]**

- [DOI](https://doi.org/10.1007/s12532-017-0119-0)
- [Author PDF](https://faculty.sist.shanghaitech.edu.cn/faculty/boris/paper/LiftedCollocation.pdf)
- Даёт: локальное lifting/condensing collocation variables и эквивалентный
  Newton/SQP step с multiple-shooting-подобной внешней структурой.
- Именно эта работа отвечает на вопрос “как применить Riccati к прямой
  коллокации”: не обязательно факторизовать полную collocation KKT как есть;
  можно локально исключить внутренние переменные интервала, а затем применить
  обычный OCP QP solver.
- Граница: требуется корректная локальная разрешимость, в частности
  невырожденность соответствующего Jacobian по внутренним collocation
  variables. При сингулярных DAE/constraints нужна более общая rank-aware
  обработка.

### 4.3. Quasi-Newton для implicit integrators

**P. Hespanhol, R. Quirynen.  
“A Real-Time Iteration Scheme with Quasi-Newton Jacobian Updates for
Nonlinear Model Predictive Control.” ECC, 2018, pp. 1517–1522.**
**[conference]**

- [DOI](https://doi.org/10.23919/ECC.2018.8550541)
- [MERL record](https://www.merl.com/publications/TR2018-082)
- Даёт: structure-preserving Jacobian updates для implicit/stiff dynamics и
  real-time iteration.

### 4.4. Decomposition и parallel linear algebra для dynamic optimization

**D. P. Word, J. Kang, J. Åkesson, C. D. Laird.  
“Efficient Parallel Solution of Large-Scale Nonlinear Dynamic Optimization
Problems.” Computational Optimization and Applications, 59(3):667–688,
2014.** **[journal]**

- [DOI](https://doi.org/10.1007/s10589-014-9651-2)
- Даёт: parallel decomposition для больших direct-transcription NLP.

**B. L. Nicholson, W. Wan, S. Kameswaran, L. T. Biegler.  
“Parallel Cyclic Reduction Strategies for Linear Systems that Arise in
Dynamic Optimization Problems.” Computational Optimization and
Applications, 70(2):321–350, 2018.** **[journal]**

- [DOI](https://doi.org/10.1007/s10589-018-0001-7)
- Даёт: cyclic reduction для block-banded KKT systems из dynamic
  optimization; альтернативный Riccati способ эксплуатации временной
  структуры.

### 4.5. Структура KKT ортогональной коллокации

**B. Ş. Cannataro, A. V. Rao, T. A. Davis.  
“State-Defect Constraint Pairing Graph Coarsening Method for KKT Matrices
Arising in Orthogonal Collocation Methods for Optimal Control.”
Computational Optimization and Applications, 64(3):793–819, 2016.**
**[journal]**

- [DOI](https://doi.org/10.1007/s10589-015-9821-x)
- [Author PDF](https://www.anilvrao.com/Publications/JournalPublications/CannataroDavisRao-COAP-July-2016.pdf)
- Даёт: graph ordering/coarsening полной KKT-матрицы коллокации для generic
  sparse factorization.
- Это другой путь, чем Riccati: сохранить полную NLP, но улучшить ordering и
  fill-in.

**M. A. Patterson, A. V. Rao.  
“Exploiting Sparsity in Direct Collocation Pseudospectral Methods for
Solving Optimal Control Problems.” Journal of Spacecraft and Rockets,
49(2), 2012.** **[journal]**

- [DOI](https://doi.org/10.2514/1.A32071)
- [Author PDF](https://anilvrao.com/Publications/JournalPublications/psStructRPM.pdf)
- Даёт: производные и sparsity exploitation для pseudospectral collocation
  NLP.

### 4.6. GPOPS-II как функциональный baseline

**M. A. Patterson, A. V. Rao.  
“GPOPS-II: A MATLAB Software for Solving Multiple-Phase Optimal Control
Problems Using hp-Adaptive Gaussian Quadrature Collocation Methods and
Sparse Nonlinear Programming.” ACM Transactions on Mathematical Software,
41(1), 2014.** **[journal/software article]**

- [DOI](https://doi.org/10.1145/2558904)
- Даёт: hp-adaptive collocation, multiple phases, static parameters и
  достаточно общую OCP-постановку.
- Граница: использует generic sparse NLP solvers; это пример требуемой
  гибкости, но не Riccati-скорости.

### 4.7. Коллокация для constrained robotics

**R. Bordalba, T. Schoels, L. Ros, J. M. Porta, M. Diehl.  
“Direct Collocation Methods for Trajectory Optimization in Constrained
Robotic Systems.” IEEE Transactions on Robotics, 39(1):183–202, 2023.**
**[journal]**

- [DOI](https://doi.org/10.1109/TRO.2022.3193776)
- [arXiv](https://arxiv.org/abs/2304.12908)
- Даёт: сравнение collocation formulations для механических систем с
  constraints.
- Граница: не предлагает универсальную Riccati factorization полной KKT.

**A. Patel, S. Shield, S. Kazi, A. M. Johnson, L. T. Biegler.  
“Contact-Implicit Trajectory Optimization Using Orthogonal Collocation.”
IEEE Robotics and Automation Letters, 4(2):2242–2249, 2019.**
**[journal]**

- [DOI](https://doi.org/10.1109/LRA.2019.2900840)
- [arXiv](https://arxiv.org/abs/1809.06436)
- Даёт: contact-implicit trajectory optimization через orthogonal
  collocation.
- Полезна как сложный application case для будущего solver, а не как источник
  Riccati-рекурсии.

### 4.8. Практический опыт адаптации коллокации к FATROP

**F. Schneider.  
“Optimal Control Parametrization Strategies for MPC with Application to
On-Orbit Servicing.” University of Stuttgart, 2025.** **[thesis]**

- [DLR record](https://elib.dlr.de/221111/)
- [PDF](https://elib.dlr.de/221111/1/Schneider-Fabio_Master_2025.pdf)
- Даёт: практическое сравнение parametrization strategies и адаптацию
  local-LGR-подобной коллокации к FATROP.
- Наблюдение: нативная collocation sparsity сама по себе не распознаётся
  стандартным OCP backend FATROP; адаптированная постановка работает, хотя в
  рассмотренных тестах multiple shooting оставался быстрее.
- Это полезное эмпирическое свидетельство, но не рецензируемая статья.

---

## 5. Неявная динамика, DAE и inverse dynamics

### 5.1. LQR с алгебраическими constraints

**J. Brüdigam, Z. Manchester.  
“Linear-Quadratic Optimal Control in Maximal Coordinates.” ICRA, 2021.**
**[conference]**

- [arXiv](https://arxiv.org/abs/2010.05886)
- Даёт: LQ optimal control в maximal coordinates с algebraic constraints и
  расширенную structured recursion.
- Граница: не является общей direct-collocation NLP formulation.

### 5.2. SQP для DAE

**B. Houska, M. Diehl.  
“A Quadratically Convergent Inexact SQP Method for Optimal Control of
Differential Algebraic Equations.” Optimal Control Applications and
Methods, 34:396–414, 2013.** **[journal]**

- [DOI](https://doi.org/10.1002/oca.2026)
- [Author PDF](https://faculty.sist.shanghaitech.edu.cn/faculty/boris/paper/inexactSQP.pdf)
- Даёт: direct multiple shooting для DAE, inexact SQP и способ не вводить
  чувствительности по большому числу algebraic states обычным образом.
- Важно: “поддержка DAE” и “прямая коллокация” — разные свойства. Неявный
  интегратор может использоваться внутри multiple shooting.

### 5.3. Inverse dynamics и локальное condensing

**S. Katayama, T. Ohtsuka.  
“Efficient Solution Method Based on Inverse Dynamics for Optimal Control
Problems of Rigid Body Systems.” arXiv:2106.04176, 2021.**
**[preprint]**

- [arXiv](https://arxiv.org/abs/2106.04176)
- Даёт: multiple-shooting formulation с inverse-dynamics equalities и
  локальным condensing.

См. также Mastalli et al., **“Inverse-Dynamics MPC via Nullspace
Resolution”**, в разделе 2.3.

### 5.4. Неявная ветка FATROP

**L. Callens.  
“Structure-exploiting LU Decomposition for Implicit Integrators in the
Riccati Recursion.”** **[repository material / internal memo]**

- Находится в ветке
  [`origin/fatropv1-implicit`](https://github.com/callenslouis/fatrop/tree/fatropv1-implicit)
  локального `fatrop_fork`.
- Локально документ можно посмотреть командой:
  `git -C fatrop_fork show origin/fatropv1-implicit:unittest/reformulation/memo/memo.tex`.
- Даёт: lifting неявного residual
  `f(x_k, u_k, x_{k+1}) = 0` через дополнительную локальную переменную и
  structure-exploiting LU внутри Riccati.
- Ветка добавляет тип неявного OCP, произвольный Jacobian по следующему
  состоянию, off-diagonal Hessian blocks, rank/preprocessing logic, тесты и
  примеры direct Radau collocation.
- По memo локальная block-LU часть может ускоряться примерно на 40–50%, а
  полный recursion — до порядка 20%; на малых задачах overhead реализации
  способен съедать выигрыш.
- Это наиболее близкая локальная заготовка для implicit collocation, но
  материал пока нельзя цитировать как опубликованную научную статью.

---

## 6. ACADO, acados и другие multistage solvers

### 6.1. ACADO Toolkit

**B. Houska, H. J. Ferreau, M. Diehl.  
“ACADO Toolkit—An Open-Source Framework for Automatic Control and Dynamic
Optimization.” Optimal Control Applications and Methods, 32(3):298–312,
2011.** **[journal]**

- [DOI](https://doi.org/10.1002/oca.939)
- [Author PDF](https://faculty.sist.shanghaitech.edu.cn/faculty/boris/paper/acado.pdf)
- Даёт: полный symbolic dynamic-optimization toolkit, включая estimation,
  control и symbolic optimization parameters.
- Важное различие: ACADO был более общим моделирующим/алгоритмическим
  toolkit; параметры можно было включать как optimization variables и
  учитывать в condensing/SQP.
- Lifted collocation в ACADO отдельно описана работой Quirynen et al. из
  раздела 4.2.

### 6.2. acados

**R. Verschueren et al.  
“acados—a Modular Open-Source Framework for Fast Embedded Optimal Control.”
Mathematical Programming Computation, 14(1):147–183, 2022.**
**[journal]**

- [DOI](https://doi.org/10.1007/s12532-021-00208-8)
- [arXiv](https://arxiv.org/abs/1910.13753)
- Даёт: modular SQP/RTI framework, direct multiple shooting, explicit и
  implicit integrators, OCP QP solvers, partial condensing и embedded code
  generation.
- Почему в обычном интерфейсе нет произвольного глобального статического
  decision vector: stagewise OCP QP ABI предполагает локальные `x_k`, `u_k`
  и соседние dynamics couplings. Поле model parameter `p` — входные данные,
  а не primal variable. Один decision block, входящий во все стадии,
  превращает KKT из block-banded в block-banded-arrow и требует изменений
  dimensions, derivative API, QP interface, condensing и backend solvers.
- Параметр можно дублировать как состояние с динамикой
  `theta_{k+1} = theta_k`, но это увеличивает state dimension на каждой стадии
  и не эквивалентно эффективной one-copy реализации.

### 6.3. Multi-phase acados

**J. Frey, K. Baumgärtner, G. Frison, M. Diehl.  
“Multi-Phase Optimal Control Problems for Efficient Nonlinear Model
Predictive Control with acados.” Optimal Control Applications and Methods,
46(2):827–845, 2025.** **[journal]**

- [DOI](https://doi.org/10.1002/oca.3234)
- [arXiv](https://arxiv.org/abs/2408.07382)
- Даёт: разные dimensions/dynamics/costs/constraints по фазам и дискретные
  phase transitions при сохранении OCP structure.
- Граница: это direct multiple shooting и не нативный произвольный
  global-static decision vector, связанный с каждой стадией. Временные
  переменные можно моделировать специальными clock states/time-speed
  состояниями, но это частный workaround.

### 6.4. qpDUNES

**J. V. Frasch, M. Vukov, H. J. Ferreau, M. Diehl.  
“A New Quadratic Programming Strategy for Efficient Sparsity Exploitation in
SQP-Based Nonlinear MPC and MHE.” IFAC Proceedings Volumes,
47(3):2945–2950, 2014.** **[conference]**

- [DOI](https://doi.org/10.3182/20140824-6-ZA-1003.01314)
- Даёт: dual Newton strategy для chain-structured OCP QP.

### 6.5. FORCES NLP

**A. Zanelli, A. Domahidi, J. L. Jerez, M. Morari.  
“FORCES NLP: An Efficient Implementation of Interior-Point Methods for
Multistage Nonlinear Nonconvex Programs.” International Journal of Control,
93(1):13–29, 2020.** **[journal]**

- [DOI](https://doi.org/10.1080/00207179.2017.1316017)
- Даёт: multistage nonlinear nonconvex IPM с генерацией специализированного
  solver.
- Граница: поддерживаемая структура задаётся multistage шаблоном; общая
  collocation/global-parameter NLP может потребовать реформулировки.

### 6.6. FORCES / multistage IPM foundation

**A. Domahidi, A. U. Zgraggen, M. N. Zeilinger, M. Morari, C. N. Jones.  
“Efficient Interior Point Methods for Multistage Problems Arising in
Receding Horizon Control.” CDC, 2012.** **[conference]**

- [EPFL record](https://infoscience.epfl.ch/entities/publication/9b4c59e1-88b9-4da7-b3da-818c46c81803)
- Даёт: structured interior-point linear algebra для multistage problems.

---

## 7. Параллельные Riccati-алгоритмы

### 7.1. Parallel Riccati implementation

**G. Frison, J. B. Jørgensen.  
“Parallel Implementation of Riccati Recursion for Solving Linear-Quadratic
Control Problems.” 18th Nordic Process Control Workshop, 2013.**
**[conference/workshop]**

- [DTU record](https://orbit.dtu.dk/en/publications/parallel-implementation-of-riccati-recursion-for-solving-linear-q/)
- Даёт: практическую параллельную декомпозицию Riccati.

### 7.2. General parallelization

**F. Laine, C. Tomlin.  
“The Parallelization of Riccati Recursion.” arXiv:1809.06360, 2018.**
**[preprint]**

- [arXiv](https://arxiv.org/abs/1809.06360)
- Даёт: ассоциативную/параллельную интерпретацию Riccati recursion.

**I. Nielsen, D. Axehill.  
“A Parallel Riccati Factorization Algorithm with Applications to Model
Predictive Control.” arXiv:1407.6898, 2014.** **[preprint]**

- [arXiv](https://arxiv.org/abs/1407.6898)
- Даёт: horizon splitting и parallel Riccati factorization для MPC.

См. также dual-regularized parallel recursion Sousa-Pinto–Orban в разделе 1.6
и parallel block-tridiagonal-arrow PIQP в разделе 3.3.

---

## 8. Что непосредственно обнаружено в этом workspace

### 8.1. `fatrop_orig`

- [README](fatrop_orig/README.md) формулирует explicit discrete OCP и ссылается
  на статью FATROP и generalized equality-constrained Riccati.
- Текущий OCP backend поддерживает stagewise equalities/inequalities и exact
  Hessian, но не one-copy global decision vector.
- В [C interface](fatrop_orig/c_interface/src/OCPCInterface.cpp) параметры
  явно отвергаются сообщением
  `Parameters are not supported anymore in the C interface`. Даже прежние
  параметры этого интерфейса были внешними данными, а не оптимизируемыми
  primal variables.
- [OCP augmented-system solver](fatrop_orig/include/fatrop/ocp/aug_system_solver.hpp)
  имеет `solve_rhs`: после одной факторизации можно решать систему с новыми
  правыми частями. Это именно тот primitive, который нужен для вычисления
  `K^{-1}B` в bordered Schur-complement алгоритме.
- В `main` есть graph backend с произвольным блочным Hessian graph и block
  Cholesky. Однако текущий graph-NLP интерфейс не покрывает нужную общую
  collocation formulation: в частности, ему недостаёт полноценной системы
  general equality constraints с межблочными связями. Поэтому просто
  представить параметры одним graph node недостаточно.

### 8.2. `fatrop_fork`

- Remote branch `origin/fatropv1-implicit` содержит активную реализацию
  implicit OCP, preprocessing/rank handling и Radau-collocation examples.
- Это наиболее подходящая база для объединения implicit dynamics/collocation
  с глобальным bordered block.
- Ветка не является частью `fatrop_orig/main`, поэтому при проектировании
  необходимо сначала отделить стабильный implicit API от экспериментального
  кода и тестов.

### 8.3. `moto`

- [README](moto/readme.md) описывает multithreaded trajectory optimizer,
  эксплуатирующий temporal и spatial sparsity implicit multiple shooting и
  использующий BLASFEO.
- [AGENTS.md](moto/AGENTS.md) и [CLAUDE.md](moto/CLAUDE.md) описывают SQP,
  nullspace/Riccati stagewise QP, IPM для inequalities, restoration/PMM и
  filter line search.
- Поле `__p` в MOTO означает **non-decision parameters**. Следовательно,
  наличие символических параметров не означает поддержку их оптимизации.
- Нативная direct collocation и один глобальный оптимизируемый parameter block
  из просмотренного интерфейса не следуют.
- Репозиторий не содержит явной библиографической ссылки на Hippo; эту связь
  нельзя фиксировать как установленный факт без подтверждения авторов.

---

## 9. Наиболее естественный алгоритм для FATROP

Для задачи с direct collocation, implicit dynamics и глобальными параметрами
наиболее экономное обобщение выглядит так:

1. **Интервальный слой коллокации.** На каждом интервале хранить state/control,
   collocation states, algebraic variables и defect residuals.
2. **Локальная факторизация или condensing.** Исключать внутренние
   collocation/algebraic unknowns либо использовать structure-exploiting LU,
   как в lifted collocation и ветке `fatropv1-implicit`.
3. **Цепочная reduced KKT.** После локального шага получить блоки, совместимые
   с OCP Riccati backend FATROP.
4. **Глобальный border.** Не размножать `theta` по стадиям; собрать его
   Hessian/Jacobian couplings в `B` и `C`.
5. **Riccati + Schur complement.** Повторно использовать факторизацию через
   `solve_rhs`, решить малую плотную систему по `theta`, затем сделать
   back-substitution.
6. **Rank и inertia handling.** Согласовать алгоритм с generalized Riccati,
   restoration phase и regularization FATROP; отдельно обработать
   singular/near-singular local collocation Jacobians.
7. **Производные.** API должен выдавать не только stage Hessians/Jacobians, но
   и смешанные блоки по `theta`, соседнему state и внутренним collocation
   variables.

Главное ограничение сложности:

$$
O\!\left(N\,c_{\text{stage}} + N\,n_\theta\,c_{\text{rhs}}
{}+ n_\theta^3\right).
$$

То есть линейность по горизонту сохраняется, пока `n_theta` невелико и не
растёт вместе с `N`.

---

## 10. Где может оставаться научная новизна

### Уже недостаточно

- просто добавить постоянный параметр как extra state;
- просто вывести bordered Schur complement;
- просто показать Riccati для switching times;
- просто поддержать fixed model parameters;
- просто локально сконденсировать regular collocation при невырожденном
  Jacobian.
- просто заметить, что local linking blocks создают sparse/banded Schur
  complement;
- просто применить hierarchical Schur decomposition к consecutive blocks.

### Потенциально сильный вклад

- единая interval-scoped factorization, в которой phase-local, interface,
  segment и mission-global decision blocks хранятся по одному разу, а
  сложность определяется максимальной активной шириной `omega`;
- совместная обработка border с rank-deficient constraints и inertia
  correction primal-dual IPM;
- implicit direct collocation без обязательного предположения, что каждый
  локальный collocation Jacobian можно устойчиво обратить;
- алгоритм, который выбирает между local condensing, lifted solve и
  uncondensed recursion;
- доказательство эквивалентности Newton step исходной collocation NLP;
- строгая оценка сложности и численной устойчивости;
- реализация в FATROP с убедительными benchmark:
  - против IPOPT на полной sparse collocation NLP;
  - против FATROP/acados на реформулированном multiple shooting;
  - против PIQP/HPIPM на QP-слое;
  - против MOTO/Hippo на робототехнических задачах;
  - с варьированием `N`, степени коллокации, числа algebraic variables и
    `n_theta`.

Рабочая формулировка возможного вклада:

> A structure-exploiting primal-dual interior-point method for nonlinear
> multiphase optimal control with implicit collocation dynamics and one-copy
> decision variables of contiguous phase scope, combining rank-aware local
> Riccati elimination with a fill-free interval-graph KKT factorization.

Она существенно точнее и защищённее, чем утверждение “первая Riccati-рекурсия
с глобальными параметрами”.

### Реализованный результат на 17 июля 2026

В `fatrop_implicit` реализован проверяемый прототип этой идеи на трёх уровнях:
fill-free interval-graph reduced KKT, полная phase-condensing схема для
explicit/implicit FATROP и точное исключение slack/complementarity в
primal-dual системе. Для интервалов фаз доказано отсутствие структурного fill
при упорядочении по правой границе и получена оценка
`O(P omega^2)` по времени и `O(P omega)` по памяти.

Полные implicit-OCP KKT решения совпадают с монолитным dense reference для
1/2/4/8 фаз и разных `(nx,nu,p)`, включая локально rank-deficient случай.
В контролируемом benchmark против оптимистичной упаковки всех статических
переменных в один dense global Schur block получено:

- почти равное время на 1--8 фазах;
- `1.30x` на 64 фазах, `1.91x` на 128 и `4.27x` на 256 при фиксированной
  `omega=8`;
- до `11.24x` на 128 фазах при росте scoped-размерностей до
  `(phase,interface,segment,global)=(8,4,8,8)`.

Максимальная невязка полного KKT в этих сериях равна `2.1e-13`; Release suite
проходит `165/165`, отдельные новые цели проходят ASan+UBSan. Добавлен
standalone nonlinear exact-Hessian Radau predictor-corrector IPM с
двусторонними неравенствами. В восьми bounded-width профилях на 1/2/4/8 фазах
он проходит независимые KKT/derivative/solution gates против того же NLP в
IPOPT+MA57 и даёт `1.00--2.55x`. Контрпример с `omega=58` честно проигрывает
MA57 (`80.023` против `66.968 ms`). Affine и corrector используют один factor:
на восьмифазном Radau-3 это уменьшило время `35.688 -> 26.340 ms`.

Это уже алгоритмический и вычислительный результат, но после учёта arrowhead
PIPS literature научная новизна остаётся кандидатной: production-интеграция,
restoration/inertia/difficult-start qualification и независимая экспертиза
claims ещё не завершены. Полная
формулировка и протокол находятся в
`fatrop_implicit/research/INTERVAL_SCOPED_RICCATI.md`.

---

## 11. Рекомендуемый порядок чтения

1. Vanroye et al. 2024 — generalized Riccati, используемая FATROP.
2. Vanroye et al. 2023 — полный nonlinear solver FATROP.
3. Quirynen et al. 2017 — lifted collocation как мост к OCP structure.
4. Schwan et al. 2025 — block-tridiagonal-arrow QP с global variables.
5. Martinez et al. 2026 — parameterized equality-constrained Riccati.
6. Callens, ветка `fatropv1-implicit` — локальная реализационная база.
7. Sousa-Pinto–Orban 2026 — regularization, inertia, cross-stage variables и
   parallel recursion.
8. Wächter–Biegler 2006 — свойства, которые нельзя потерять относительно
   IPOPT.
9. Rehfeldt et al. 2022 и Kempke et al. 2026 — local-linking arrowhead IPM и
   ближайшая generic граница новизны.
10. Kang et al. 2014 — общий nonlinear Schur-complement взгляд.
11. Hippo 2026 и MOTO — современный робототехнический stagewise solver
    baseline.
