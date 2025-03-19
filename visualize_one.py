def visualize_one(image, prediction, threshold=0.5):

    ax = plt.gca()
    box = prediction[0].boxes.xyxy[0].cpu().numpy()

    img_array = np.array(image, dtype=int)
    isolated_face = img_array[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
    isolated_face = np.uint8(isolated_face)
    face_img = Image.fromarray(isolated_face)
    
    
    plt.axis("off")
    #plt.figure(figsize=(12, 8))
    plt.imshow(face_img)
    plt.show()
