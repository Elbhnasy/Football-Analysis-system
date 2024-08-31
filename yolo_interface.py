from ultralytics import YOLO

# Load the model
model = YOLO('/home/fox/Desktop/Football-Analysis-system/models/v5.pt')

# Predict the results on a video
results = model.predict('videos/08fd33_4.mp4', save = True)  

# Print the results
print(results[0])

print ('===============================')

# Print the boxes
for box in results[0].boxes:
    print(box)
    