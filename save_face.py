def save_face(image, prediction, filename, threshold=0.5):

    ax = plt.gca()
    box = prediction[0].boxes.xyxy[0].cpu().numpy()

    img_array = np.array(image, dtype=int)
    isolated_face = img_array[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
    isolated_face = np.uint8(isolated_face)
    face_img = Image.fromarray(isolated_face)
    
    ######tego nie trzeba#######
        # plt.axis("off")
        # plt.imshow(face_img)
        # plt.show()
    ############################
    
    face_img.save(filename)

#image - zdjecie ( obiekt PIL)
#prediction - zdjecie przepuszczone przez yolo - np. yolo(image) lub model(image)
#filename - nazwa, pod jaka chcecie zapisac zdjecie (sciezka do pliku)


#####    NA PRZYKLAD    #######

image = Image.open('jude_law.jpg').convert('RGB')
prediction = model(image)
save_face(image, prediction, 'twarze/twarz.jpg')
