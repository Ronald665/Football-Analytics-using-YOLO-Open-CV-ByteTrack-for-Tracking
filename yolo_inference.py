from ultralytics import YOLO

model = YOLO('Modules/best.pt')

results = model.predict('Input _videos/08fd33_4.mp4', save=True)
print(results[0])

print('============================')
for box in results.boxes:
    print(box)