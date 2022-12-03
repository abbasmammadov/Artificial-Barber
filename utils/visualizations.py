import os
from matplotlib import pyplot as plt
im_path1 = 'img/appearance'
im_path2 = 'img/structure'
im_names1 = ['red.png', 'rainbow.png', 'blue.png']
im_names2 = ['bob.png', 'fade.png', 'straight.png']

# lets make a grid of 2*3
fig = plt.figure(figsize=(10, 5))
# lets plot the first row
for i in range(3):
    ax = fig.add_subplot(2, 3, i+1)
    # lets add img names without suffix to the title

    ax.imshow(plt.imread(os.path.join(im_path1, im_names1[i])))
    ax.set_title(im_names1[i].split('.')[0])

    ax.axis('off')

# lets plot the second row
for i in range(3):
    ax = fig.add_subplot(2, 3, i+4)
    ax.imshow(plt.imread(os.path.join(im_path2, im_names2[i])))
    ax.set_title(im_names2[i].split('.')[0])
    ax.axis('off')

# lets save the figure
fig.savefig('img/row_col.png')