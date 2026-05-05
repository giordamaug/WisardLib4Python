import random
import numpy as np
import numpy as np
from collections import Counter
from tqdm.notebook import tqdm
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
from sklearnapi import WiSARDClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
import sklearn.metrics as metrics

# =========================
# Fitness function
# =========================
def evaluate_individual_cv(mapping, X, y, n_bits, n_tics, seed):
    clf = WiSARDClassifier(len(X[0]), n_bits=n_bits, n_tics=n_tics, n_classes=len(np.unique(y)), random_state=-1, # explicit input mapping
                           mapping=mapping.ravel(), bleaching=False, code='t', debug=False)
    y_pred = cross_val_predict(clf, X, y, cv=5)
    return metrics.accuracy_score(y, y_pred)

# =========================
# Crossover operator
# =========================

def find_duplicates(matrix):
    flat = matrix.flatten()
    counts = Counter(flat)
    
    # duplicati: includiamo tante copie quante sono in eccesso
    duplicates = []
    for val, c in counts.items():
        if c > 1:
            duplicates.extend([val] * (c - 1))
    
    return sorted(duplicates)


def replace_first_occurrence(matrix, value, new_value):
    """Sostituisce la prima occorrenza trovata (scansione row-wise)"""
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == value:
                matrix[i, j] = new_value
                return


def fix_by_swapping(child1, child2):
    d1 = find_duplicates(child1)
    d2 = find_duplicates(child2)

    if len(d1) != len(d2):
        raise ValueError("Le liste duplicati devono avere stessa lunghezza")

    # swap posizione-per-posizione
    for v1, v2 in zip(d1, d2):
        replace_first_occurrence(child1, v1, v2)
        replace_first_occurrence(child2, v2, v1)

    return child1, child2


def crossover(Mx, My, crossover_row=4):
    Mx = Mx.copy()
    My = My.copy()

    # Step 1: crossover
    child1 = np.vstack((Mx[:crossover_row], My[crossover_row:]))
    child2 = np.vstack((My[:crossover_row], Mx[crossover_row:]))

    # Step 2: fixing con swap duplicati
    child1, child2 = fix_by_swapping(child1, child2)

    return child1, child2
    
# =========================
# Mutate operator
# =========================

def mutate(M, debug=False):
    M = M.copy()
    m, n = M.shape

    factor = 4

    # numero di coppie
    n_pairs = m // factor

    if debug:
        print("n_pairs:", n_pairs)

    # seleziona 2*n_pairs righe distinte
    selected_rows = random.sample(range(m), 2 * n_pairs)

    # 🔥 shuffle per rendere le coppie RANDOMICHE
    random.shuffle(selected_rows)

    # crea coppie NON sovrapposte
    pairs = [(selected_rows[2*i], selected_rows[2*i+1]) for i in range(n_pairs)]

    # swap ciclico sulle colonne
    for i, (r1, r2) in enumerate(pairs):
        col = i % n
        M[r1, col], M[r2, col] = M[r2, col], M[r1, col]

    return M

def mutate_1d_np(ind, m, n, factor=4):
    n_pairs = m // factor

    rows = np.random.choice(m, size=2*n_pairs, replace=False)
    np.random.shuffle(rows)

    r1 = rows[0::2]
    r2 = rows[1::2]
    cols = np.arange(n_pairs) % n

    idx1 = r1 * n + cols
    idx2 = r2 * n + cols

    ind[idx1], ind[idx2] = ind[idx2].copy(), ind[idx1].copy()

    return ind

# =========================
# Utility
# =========================

def clone(ind):
    return ind.copy()


def random_pair(pop):
    return random.sample(pop, 2)


def random_individual(pop):
    return random.choice(pop)


# =========================
# Inizializzazione
# =========================

def initialize(mu, m, n, linear=True):
    population = []
    base = np.arange(m * n)

    if not linear:
        for _ in range(mu):
            perm = np.random.permutation(base)
            population.append(perm.reshape(m, n))
    else:
        for _ in range(mu):
            population.append(base.reshape(m, n))

    return population


# =========================
# Evaluate (DA DEFINIRE)
# =========================

def evaluate(population, X, y, n_bits, n_tics, seed):
    return Parallel(n_jobs=multiprocessing.cpu_count())(
        #delayed(evaluate_individual)(ind, X, y, n_bits, n_tics, seed) for ind in population
        delayed(evaluate_individual_cv)(ind, X, y, n_bits, n_tics, seed) for ind in population
    )

# =========================
# Selection (μ best)
# =========================

def select(population, fitness, mu, maximize=True):
    sorted_idx = np.argsort(fitness)
    if maximize:
        sorted_idx = sorted_idx[::-1]
    return [population[i] for i in sorted_idx[:mu]]

# =========================
# MAIN EA (μ + λ)
# =========================

def ea_mu_lambda(mu, lam, tau, theta_r, theta_m, m, n, 
                 X, y, n_bits, n_tics, seed,
                 crossover_fn, mutate_fn,
                 patience=20, min_delta=1e-4, maximize=True, 
                 live_plot=True):

    # init
    population = initialize(mu, m, n)
    fitness = evaluate(population, X, y, n_bits, n_tics, seed)

    best_fitness = -np.inf
    patience_counter = 0

    history = {"min": [], "max": [], "mean": []}
    if live_plot:
        fig, ax = plt.subplots()
        display_handle = display(fig, display_id=True)

    outer_bar = tqdm(range(tau), desc="Generations")

    for t in outer_bar:

        offspring = []

        for _ in tqdm(range(lam), desc="Offspring", leave=False):
            choice = random.random()

            if choice < theta_r:
                p1, p2 = random_pair(population)
                c1, c2 = crossover_fn(clone(p1), clone(p2))
                offspring.append(c1)

            elif choice < theta_r + theta_m:
                p = random_individual(population)
                child = mutate_fn(clone(p))
                offspring.append(child)

            else:
                p = random_individual(population)
                offspring.append(clone(p))

        # merge
        population = population + offspring

        # evaluate
        fitness = evaluate(population, X, y, n_bits, n_tics, seed)

        # select
        population = select(population, fitness, mu, maximize=maximize)

        # fitness coerente
        fitness = evaluate(population, X, y, n_bits, n_tics, seed)

        # stats
        f_min = np.min(fitness)
        f_max = np.max(fitness)
        f_mean = np.mean(fitness)

        history["min"].append(f_min)
        history["max"].append(f_max)
        history["mean"].append(f_mean)

        # ---- EARLY STOPPING ----
        if f_max > best_fitness + min_delta:
            best_fitness = f_max
            patience_counter = 0
        else:
            patience_counter += 1

        # tqdm update
        outer_bar.set_postfix({
            "min": f"{f_min:.4f}",
            "max": f"{f_max:.4f}",
            "mean": f"{f_mean:.4f}",
            "pat": patience_counter
        })

        # ---- LIVE PLOT ----
        if live_plot: #and t % 5 == 0:
            ax.clear()

            x = range(len(history["max"]))

            ax.plot(x, history["max"], label="max")
            ax.plot(x, history["mean"], label="mean")
            ax.plot(x, history["min"], label="min")

            ax.set_title(f"Fitness Evolution (gen {t})")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Fitness")
            ax.legend()

            display_handle.update(fig)

        # stop
        if patience_counter >= patience:
            print(f"\nEarly stopping at generation {t}")
            break

    return population, history

def ea_mu_lambda_old(mu, lam, tau, theta_r, theta_m, m, n, evaluate, 
                 crossover_fn, mutate_fn):

    # init
    population = initialize(mu, m, n)
    fitness = evaluate(population)

    # progress bar esterna (generazioni)
    for t in tqdm(range(tau), desc="Generations"):

        offspring = []

        # progress bar interna (lambda offspring)
        for _ in tqdm(range(lam), desc="Offspring", leave=False):
            choice = random.random()

            if choice < theta_r:
                # crossover
                p1, p2 = random_pair(population)
                c1, c2 = crossover_fn(clone(p1), clone(p2))
                offspring.append(c1)

            elif choice < theta_r + theta_m:
                # mutation
                p = random_individual(population)
                child = mutate_fn(clone(p))
                offspring.append(child)

            else:
                # reproduction
                p = random_individual(population)
                offspring.append(clone(p))

        # merge (μ + λ)
        population = population + offspring

        # evaluate
        fitness = evaluate(population)

        # select best μ
        population = select(population, fitness, mu)

        # opzionale: aggiorna fitness coerente
        fitness = evaluate(population)

    return population