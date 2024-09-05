from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Büyük kutunun boyutları (1016 x 1219.2 x 2700)
container_dims = [1016, 1219.2, 2700]

# Kalan alanı hesaplayan fonksiyon
def calculate_remaining_space(container_dims, placed_boxes):
    """Mevcut yerleştirilen kutulardan sonra kalan alanı hesaplar."""
    used_volume = sum(dim[0] * dim[1] * dim[2] for _, dim, _ in placed_boxes)
    total_volume = container_dims[0] * container_dims[1] * container_dims[2]
    remaining_volume = total_volume - used_volume
    return remaining_volume

# Greedy algoritmasını tanımlayan fonksiyon
def greedy_place_box(container_dims, orientations, trials=100):
    """Kutu yerleştirme işlemi - Gelişmiş Greedy Algoritma"""
    best_placed_boxes = []
    best_remaining_space = float('inf')
    
    for i in range(trials):
        placed_boxes = []
        for orientation, name in orientations:
            for x_pos in np.arange(0, container_dims[0] - orientation[0] + 1, orientation[0]):
                for y_pos in np.arange(0, container_dims[1] - orientation[1] + 1, orientation[1]):
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

        remaining_space = calculate_remaining_space(container_dims, placed_boxes)
        
        # Eğer mevcut denemede daha az boş alan kalmışsa bu çözümü seçiyoruz
        if remaining_space < best_remaining_space:
            best_remaining_space = remaining_space
            best_placed_boxes = placed_boxes[:]

    return best_placed_boxes

# Rotasyonları dene ve en iyi çözümü bul
def find_best_rotation(container_dims, x, y, z):
    """Tüm rotasyonları dener ve en iyi kutu yerleşimini bulur."""
    orientations = [
        ((x, y, z), "xyz"),
        ((x, z, y), "xzy"),
        ((y, x, z), "yxz"),
        ((y, z, x), "yzx"),
        ((z, x, y), "zxy"),
        ((z, y, x), "zyx"),
    ]
    
    best_solution = []
    best_remaining_space = float('inf')
    
    # Her rotasyonu dene
    for orientation in orientations:
        placed_boxes = greedy_place_box(container_dims, [orientation], trials=100)
        remaining_space = calculate_remaining_space(container_dims, placed_boxes)
        
        if remaining_space < best_remaining_space:
            best_remaining_space = remaining_space
            best_solution = placed_boxes
            
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

@app.route('/')
def index():
    return render_template('index.html')

# Bu kısım düzeltilmiş
@app.route('/calculate', methods=['POST'])
def calculate():
    x = float(request.form['x'])
    y = float(request.form['y'])
    z = float(request.form['z'])

    # En iyi rotasyonu bul
    best_solution = find_best_rotation(container_dims, x, y, z)

    # Kutuların adedini hesapla
    number_of_boxes = len(best_solution)

    # 3D grafiği çiz ve görselleştir
    img_data = plot_solution(best_solution)

    # Sonuçları şablona gönder
    return render_template('result.html', img_data=img_data, number_of_boxes=number_of_boxes)

if __name__ == '__main__':
    app.run(debug=True)

