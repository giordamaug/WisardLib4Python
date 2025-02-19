import numpy as np
import wisard

# Inizializza il classificatore WiSARD
model = wisard.WiSARD(64, n_bits=2, classes=[0, 1])

# Dati di esempio (immagine binaria)
X = np.random.randint(0, 2, size=(64,), dtype=np.uint8)
y = 1  # Etichetta

print(X)

print(model._mk_tuple(X))
# Allenamento
model.train(X, 1)

# Test
result = model.test(X, 0)
print(f"Predizione: {result}")

# Risposta dettagliata
response = model.response(X, threshold=0, percentage=True)
print(f"Risultati: {response}")

print(f"Mental Image: {model.getMI(1)})
