import random
import numpy as np
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Büyük kutunun boyutları (1016 x 1219.2 x 2700)
container_dims = [1016, 1219.2, 2700]

# Kalan alanı hesaplayan fonksiyon
def calculate_remaining_space(container_dims, placed_boxes):
    used_volume = sum(dim[0] * dim[1] * dim[2] for _, dim, _ in placed_boxes)
    total_volume = container_dims[0] * container_dims[1] * container_dims[2]
    remaining_volume = total_volume - used_volume
    return remaining_volume

# Greedy algoritması başlangıç popülasyonu için
def greedy_place_box(container_dims, box_dims):
    """Kutuları farklı rotasyonlarla yerleştirip en iyi çözümü bulur."""
    placed_boxes = []

    # Z eksenine asla orientation[2] koymadan kullanılan rotasyonlar
    orientations = [
        ((box_dims[0], box_dims[2], box_dims[1]), "xzy"),  
        ((box_dims[1], box_dims[2], box_dims[0]), "yzx"),  
        ((box_dims[2], box_dims[0], box_dims[1]), "zxy"),  
        ((box_dims[2], box_dims[1], box_dims[0]), "zyx"),
    ]
    
    for orientation, name in orientations:
        for x_pos in np.arange(0, container_dims[0] - orientation[0] + 1, 1):
            for y_pos in np.arange(0, container_dims[1] - orientation[1] + 1, 1):
                for z_pos in np.arange(0, container_dims[2] - orientation[2] + 1, orientation[2]):
                    overlap = False
                    for pos, dim, _ in placed_boxes:
                        if (x_pos < pos[0] + dim[0] and x_pos + orientation[0] > pos[0] and
                            y_pos < pos[1] + dim[1] and y_pos + orientation[1] > pos[1] and
                            z_pos < pos[2] + dim[2] and z_pos + orientation[2] > pos[2]):
                            overlap = True
                            break
                    if not overlap:
                        placed_boxes.append(((x_pos, y_pos, z_pos), orientation, name))
                        break

    return placed_boxes

# Fitness fonksiyonu (boş alanı minimize etme üzerine kurulu)
def fitness_function(container_dims, placed_boxes):
    return -calculate_remaining_space(container_dims, placed_boxes)

# Popülasyonun başlangıçta greedy algoritma ile oluşturulması
def create_initial_population(container_dims, box_dims, population_size=10):
    population = []
    for _ in range(population_size):
        solution = greedy_place_box(container_dims, box_dims)
        population.append(solution)
    return population

# İki çözümü çaprazlama (crossover) işlemi
def crossover(parent1, parent2):
    crossover_point = random.randint(0, min(len(parent1), len(parent2)) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

# Mutasyon işlemi (çözümde küçük değişiklikler)
def mutate(solution, mutation_rate=0.1):
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            # Rastgele bir pozisyonu değiştir
            solution[i] = (solution[i][0], solution[i][1], solution[i][2])
    return solution

# Genetik algoritma ile popülasyonu evrimleştirme
def genetic_algorithm(container_dims, box_dims, generations=50, population_size=10):
    population = create_initial_population(container_dims, box_dims, population_size)
    
    for _ in range(generations):
        # Fitness hesaplama ve sıralama
        population = sorted(population, key=lambda p: fitness_function(container_dims, p), reverse=True)
        
        # Seçilen en iyi çözümleri yeni popülasyona ekle
        new_population = population[:2]  # En iyi iki çözümü tut
        
        # Çaprazlama ve mutasyonla yeni çözümler üret
        while len(new_population) < population_size:
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        
        population = new_population
    
    # En iyi çözümü geri döndür
    best_solution = sorted(population, key=lambda p: fitness_function(container_dims, p), reverse=True)[0]
    return best_solution

# 3D görselleştirme fonksiyonu
def plot_solution(placed_boxes):
    """Yerleştirilen kutuları 3D olarak görselleştirir"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim([0, container_dims[0]])
    ax.set_ylim([0, container_dims[1]])
    ax.set_zlim([0, container_dims[2]])

    # Kutulara sırayla siyah ve beyaz renk atıyoruz
    colors = ['white', 'black']
    for idx, (pos, dim, _) in enumerate(placed_boxes):
        color = colors[idx % 2]  # Bir siyah bir beyaz
        ax.bar3d(pos[0], pos[1], pos[2], dim[0], dim[1], dim[2], shade=True, color=color)

    ax.set_xlabel('X Eksen')
    ax.set_ylabel('Y Eksen')
    ax.set_zlabel('Z Eksen')

    # Resmi kaydet ve base64 formatında döndür
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode('ascii')
    return img

# Kutuların ne kadar yer kapladığını yüzdesel olarak hesaplama
def calculate_fill_percentage(container_dims, placed_boxes):
    total_volume = container_dims[0] * container_dims[1] * container_dims[2]
    used_volume = sum(dim[0] * dim[1] * dim[2] for _, dim, _ in placed_boxes)
    fill_percentage = (used_volume / total_volume) * 100
    return fill_percentage

@app.route('/')
def index():
    return render_template('index.html')

# Flask üzerinden kutu yerleştirme işlemi başlatma
@app.route('/calculate', methods=['POST'])
def calculate():
    x = float(request.form['x'])
    y = float(request.form['y'])
    z = float(request.form['z'])

    # Genetik algoritma ile en iyi çözümü bul
    best_solution = genetic_algorithm(container_dims, [x, y, z])

    # Kutuların adedini hesapla
    number_of_boxes = len(best_solution)

    # Konteynerin ne kadar dolduğunu yüzdesel olarak hesapla
    fill_percentage = calculate_fill_percentage(container_dims, best_solution)

    # 3D grafiği çiz ve görselleştir
    img_data = plot_solution(best_solution)

    # Sonuçları şablona gönder
    return render_template('result.html', img_data=img_data, number_of_boxes=number_of_boxes, fill_percentage=fill_percentage)

if __name__ == '__main__':
    app.run(debug=True)
