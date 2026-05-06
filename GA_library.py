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
import sklearn.base as skbase

# =========================
# paralle evaluation of fitness
# =========================


def evaluate(population, X, y, clf_params):
    return Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(evaluate_individual_cv)(ind, X, y, clf_params) for ind in population
    )

# =========================
# Specific Fitness function
# =========================

def evaluate_individual_cv(ind, X, y, clf_params):
    clf = WiSARDClassifier(**clf_params, mapping=ind.ravel(), random_state=-1)
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

def crossover(ind1, ind2, m, n, crossover_row=None):
    if crossover_row is None:
        crossover_row = np.random.randint(1, m)
    ind1 = ind1.copy()
    ind2 = ind2.copy()

    # Step 1: crossover
    child1 = np.vstack((ind1[:crossover_row], ind2[crossover_row:]))
    child2 = np.vstack((ind2[:crossover_row], ind1[crossover_row:]))

    # Step 2: fixing con swap duplicati
    child1, child2 = fix_by_swapping(child1, child2)

    return child1, child2
    
def find_duplicates_1d_np(vec):
    counts = Counter(vec)

    duplicates = []
    for val, c in counts.items():
        if c > 1:
            duplicates.extend([val] * (c - 1))
    return sorted(duplicates)

def replace_first_occurrence_1d_np(vec, value, new_value):
    idx = np.flatnonzero(vec == value)
    if len(idx) > 0:
        vec[idx[0]] = new_value

def fix_by_swapping_1d_np(child1, child2):
    d1 = find_duplicates_1d_np(child1)
    d2 = find_duplicates_1d_np(child2)

    if len(d1) != len(d2):
        raise ValueError("Le liste duplicati devono avere stessa lunghezza")

    # swap posizione-per-posizione
    for v1, v2 in zip(d1, d2):
        replace_first_occurrence_1d_np(child1, v1, v2)
        replace_first_occurrence_1d_np(child2, v2, v1)

    return child1, child2

def crossover_1d_np(ind1, ind2, m, n, crossover_row=None):
    if crossover_row is None:
        crossover_row = np.random.randint(1, m)

    k = crossover_row * n

    child1 = np.empty_like(ind1)
    child2 = np.empty_like(ind2)

    child1[:k] = ind1[:k]
    child1[k:] = ind2[k:]

    child2[:k] = ind2[:k]
    child2[k:] = ind1[k:]

    fix_by_swapping_1d_np(child1, child2)

    ind1[:] = child1
    ind2[:] = child2

    return ind1, ind2

# =========================
# Mutate operator
# =========================

def mutate(ind, m, n, factor=4):
    ind = ind.copy()

    # numero di coppie
    n_pairs = m // factor

    # seleziona 2*n_pairs righe distinte
    selected_rows = random.sample(range(m), 2 * n_pairs)

    # 🔥 shuffle per rendere le coppie RANDOMICHE
    random.shuffle(selected_rows)

    # crea coppie NON sovrapposte
    pairs = [(selected_rows[2*i], selected_rows[2*i+1]) for i in range(n_pairs)]

    # swap ciclico sulle colonne
    for i, (r1, r2) in enumerate(pairs):
        col = i % n
        ind[r1, col], ind[r2, col] = ind[r2, col], ind[r1, col]

    return ind

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

def initialize_1d_np(mu, m, n, linear=True):
    population = []
    base = np.arange(m * n)

    if not linear:
        for _ in range(mu):
            perm = np.random.permutation(base)
            population.append(perm)
    else:
        for _ in range(mu):
            population.append(base)

    return population


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
                 X, y,
                 clf_params,
                 crossover_fn, mutate_fn, initialize_fn,
                 patience=20, min_delta=1e-4, maximize=True, start_with_linear=True,
                 live_plot=True):

    # init
    population = initialize_fn(mu, m, n, linear=start_with_linear)
    fitness = evaluate(population, X, y, clf_params)

    best_fitness = -np.inf
    patience_counter = 0

    history = {"min": [], "max": [], "mean": [], "crossover": [], "mutation": [], "clone": [],}
    if live_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        fig.subplots_adjust(hspace=0.4)
        display_handle = display(fig, display_id=True)

    outer_bar = tqdm(range(tau), desc="Generations")

    for t in outer_bar:

        offspring = []
        n_crossover = 0
        n_mutation = 0
        n_clone = 0

        for _ in tqdm(range(lam), desc="Offspring", leave=False):
            choice = random.random()

            if choice < theta_r:
                p1, p2 = random_pair(population)
                c1, c2 = crossover_fn(clone(p1), clone(p2), m, n, crossover_row=int(m/2))
                offspring.append(c1)
                n_crossover += 1

            elif choice < theta_r + theta_m:
                p = random_individual(population)
                child = mutate_fn(clone(p), m, n)
                offspring.append(child)
                n_mutation += 1

            else:
                p = random_individual(population)
                offspring.append(clone(p))
                n_clone += 1

        # merge
        population = population + offspring

        # evaluate
        fitness = evaluate(population, X, y, clf_params)

        # select
        population = select(population, fitness, mu, maximize=maximize)

        # fitness coerente
        fitness = evaluate(population, X, y, clf_params)

        # stats
        f_min = np.min(fitness)
        f_max = np.max(fitness)
        f_mean = np.mean(fitness)

        history["min"].append(f_min)
        history["max"].append(f_max)
        history["mean"].append(f_mean)
        history["crossover"].append(n_crossover)
        history["mutation"].append(n_mutation)
        history["clone"].append(n_clone)

        # ---- EARLY STOPPING ----
        if f_max > best_fitness + min_delta:
            best_fitness = f_max           # or f_mean
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
            ax1.clear()
            ax2.clear()

            x = np.arange(len(history["max"]))

            # ---- FITNESS ----
            ax1.plot(x, history["max"], label="max")
            ax1.plot(x, history["mean"], label="mean")
            ax1.plot(x, history["min"], label="min")

            ax1.set_title(f"Fitness Evolution (gen {t})")
            ax1.set_xlabel("Generation")
            ax1.set_ylabel("Fitness")
            ax1.legend()
            ax1.grid(True)

            # ---- OPERATORI ----
            width = 0.25

            ax2.bar(x - width, history["crossover"], width=width, label="crossover", alpha=0.6)
            ax2.bar(x, history["mutation"], width=width, label="mutation", alpha=0.6)
            ax2.bar(x + width, history["clone"], width=width, label="clone", alpha=0.6)

            ax2.set_title("Operator Usage per Generation")
            ax2.set_xlabel("Generation")
            ax2.set_ylabel("Count")
            ax2.legend()
            ax2.grid(True)

            display_handle.update(fig)

        # stop
        if patience_counter >= patience:
            print(f"\nEarly stopping at generation {t}")
            break

    return population, history

## OLD stuff
def evaluate_old(population, X, y, n_bits, n_tics, seed):
    return Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(evaluate_individual_cv_old)(ind, X, y, n_bits, n_tics, seed) for ind in population
    )

def evaluate_individual_cv_old(mapping, X, y, n_bits, n_tics, seed):
    clf = WiSARDClassifier(len(X[0]), n_bits=n_bits, n_tics=n_tics, n_classes=len(np.unique(y)), random_state=-1, # explicit input mapping
                           mapping=mapping.ravel(), bleaching=False, code='t', debug=False)
    y_pred = cross_val_predict(clf, X, y, cv=5)
    return metrics.accuracy_score(y, y_pred)
