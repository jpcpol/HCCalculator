import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Cargar el modelo pre-entrenado y el mapa de etiquetas
PATH_TO_MODEL = 'path_to_your_pretrained_model'
PATH_TO_LABELS = 'path_to_your_label_map'
NUM_CLASSES = 90

# Cargar el modelo pre-entrenado
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Cargar el mapa de etiquetas
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Función para detectar objetos en la imagen
def detect_objects(image):
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            # Definir las operaciones de entrada y salida del grafo de detección
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
            # Expandir las dimensiones de la imagen para que coincidan con el formato esperado por el modelo
            image_expanded = np.expand_dims(image, axis=0)
            
            # Realizar la detección
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_expanded})
            
            # Visualizar los resultados de la detección
            vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            
            return image, boxes, scores, classes, num

# Cargar la imagen
image_path = 'path_to_your_image.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convertir la imagen a escala de grises
imagen_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar un umbral para obtener una imagen binaria
_, imagen_binaria = cv2.threshold(imagen_gris, 127, 255, cv2.THRESH_BINARY)

# Encontrar contornos en la imagen binaria
contornos, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar los contornos en la imagen original
cv2.drawContours(image, contornos, -1, (0, 255, 0), 2)

# Detectar objetos en la imagen
image_with_boxes, boxes, scores, classes, num = detect_objects(image_rgb)

# Definir una base de datos de alimentos con su contenido de carbohidratos (en gramos)
food_database = {
    'apple': 25,
    'banana': 27,
    'orange': 12,
    # Agregar más alimentos según sea necesario
}
# Estimar la cantidad de carbohidratos en cada alimento detectado
total_carbohydrates = 0
for score, class_id, box in zip(np.squeeze(scores), np.squeeze(classes).astype(np.int32), np.squeeze(boxes)):
    if score > 0.5:  # Umbral de confianza
        class_name = category_index[class_id]['name']
        if class_name.lower() in food_database:
            carbohydrates = food_database[class_name.lower()]
            total_carbohydrates += carbohydrates

# Mostrar la imagen con los objetos detectados y el total de carbohidratos estimado
cv2.putText(image_with_boxes, f'Total de carbohidratos: {total_carbohydrates}g', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
cv2.imshow('Detected Objects', cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()