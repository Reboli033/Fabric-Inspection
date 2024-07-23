import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time


def read_image(file_path):
    image = Image.open(file_path)
    if image is None:
        print(f"Error: Could not read the image from {file_path}.")
    return np.array(image)


# Image preprocessing
def preprocess_image(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    unsharp_masked_image = cv2.addWeighted(gray_image, 1.5, blurred_image, -0.5, 0)
    # bilateral_filtered_image = cv2.bilateralFilter(gray_image, d=9, sigmaColor=75, sigmaSpace=75)

    return unsharp_masked_image


# Sobel edge detection for both X and Y
def sobel_edge_detection(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    return sobelx, sobely


def extract_horizontal_portion(binary_image, start_y, end_y):
    # Ensure binary_image is a binary image (values are 0 or 255)
    _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

    # Get the height and width of the image
    height, width = binary_image.shape[:2]

    # Create a mask with the desired horizontal portion
    mask = np.zeros_like(binary_image)
    mask[start_y:end_y, :] = 255

    # Apply the mask to the binary image
    horizontal_portion = cv2.bitwise_and(binary_image, mask)

    return horizontal_portion

def cal_original_length(straighten_yarn_length, yarn_length_fabric,needed):
    crimp_per = ((straighten_yarn_length - yarn_length_fabric)/yarn_length_fabric) * 100

    #crimp_for_needed = (crimp_per/yarn_length_fabric) * needed
    original_length = (1 + (crimp_per))* needed
    #(straighten_yarn_length/yarn_length_fabric)*needed #single yarn

    return original_length

# Main function
def main():
    # Specify the path to your image file
    image_path = r"C:\Users\Dell\PycharmProjects\pythonProject1\final2.png"  # Replace with the actual path

    # Read image from file
    image = read_image(image_path)

    if image is not None:
        # Preprocess the image
        preprocessed_image = preprocess_image(image)
        i = Image.fromarray(preprocessed_image)
        i.show()

        # Sobel edge detection for both X and Y

        sobelx, sobely = sobel_edge_detection(preprocessed_image)
        x = Image.fromarray(sobelx)
        #x.show()
        y = Image.fromarray(sobely)
        #y.show()

        kernel = np.ones((4, 4), np.uint8)
        #kernel2 = np.ones((5, 5), np.uint8)

        # Apply morphological opening (erosion followed by dilation) to reduce noise
        opened_image_X = cv2.morphologyEx(sobelx, cv2.MORPH_OPEN, kernel)
        opened_image_y = cv2.morphologyEx(sobely, cv2.MORPH_OPEN, kernel)

        # Display the results
        #cv2.imshow('Original Sobelx', sobelx)
        #cv2.imshow('Opened Image_X', opened_image_X)
        cv2.imshow('Opened Image_y', opened_image_y)
        cv2.waitKey(0)

        cv2.destroyAllWindows()
        ##sobel X #########################################################################################################################
        image_array = []

        for j in range(0, 6):
            # Define the start and end y-coordinates for the horizontal portion
            start_y = 0 + j * 50
            end_y = 10 + j * 50

            # Extract the horizontal portion
            result = extract_horizontal_portion(opened_image_X, start_y, end_y)

            image_array.append(result)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        # print(len(image_array))
        for j in image_array:
            img = Image.fromarray(j)
            #img.show()
            cv2.waitKey(0)
            cv2.destroyAllWindows()




        x_at_y_outer = []

        for image_index in range(0, 6):
            x_at_y_array = []
            for j in range(0, 601):
                y = 0 + image_index * 50
                pixel_value = image_array[image_index][y, j]
                #print(f" ({j},{y}): {pixel_value},")
                x_at_y_array.append(pixel_value)
            x_at_y_outer.append(x_at_y_array)

        x_array = list(range(0, 601))

       # plot x_at_50
        """darkened_color = (0.2, 0.2, 0.2)
        transparency = 0.6
        plt.plot(x_array[0:50],x_at_y_outer[0][0:50],color = darkened_color, label = 'Sample 1',alpha = transparency)
        #plt.xlabel("X coordinates")
        #plt.ylabel("Pixel Values")
        #plt.show()
        plt.plot(x_array, x_at_y_outer[1],color = darkened_color,label = 'Sample 2',alpha = transparency)
        plt.plot(x_array, x_at_y_outer[2],color = darkened_color,label = 'Sample 3',alpha = transparency)

        plt.plot(x_array, x_at_y_outer[3],color = darkened_color,label = 'Sample 4',alpha = transparency)

        plt.plot(x_array, x_at_y_outer[4],color = darkened_color, label='Sample 5',alpha = transparency)

        plt.plot(x_array, x_at_y_outer[5],color = darkened_color, label='Sample 6',alpha = transparency)


        plt.xlabel("X coordinates")
        plt.ylabel("Pixel Values")
        plt.legend()
        plt.show()"""
        #print(f"x co-ordinates of the image \n")
        """print(x_array[0:15], "\n")
        
        for i in range(0,6):
            print(f"Sample{i+1} sequence:")
            print(x_at_y_outer[i][0:15],"\n")"""

        #calculate yarn count
        yarn_count1 = 0
        for i in range(0, 601):
            count = 0
            for j in range(0,6):
                #print(x_at_y_outer[j][i])
                if x_at_y_outer[j][i] == 255.0:
                    count = count + 1
            #print(f"count at {i}:{count}")
            if count == 6:
                yarn_count1 = yarn_count1 + 1
        #print(f"warps count:{yarn_count1}")

        ###########################################################################################################################


        #################sobel Y #################################################################################################

        """image_array2 = []

        for j in range(0, 6):
            # Define the start and end y-coordinates for the horizontal portion
            start_x = 0 + j * 50
            end_x = 10 + j * 50

            # Extract the horizontal portion
            result = extract_horizontal_portion(opened_image_y, start_x, end_x)

            image_array2.append(result)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        # print(len(image_array))
        for j in image_array2:
            img = Image.fromarray(j)
            img.show()
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        y_at_x_outer = []

        for image_index in range(0, 6):
            y_at_x_array = []
            for j in range(0, 302):
                x = 0 + image_index * 50
                pixel_value = image_array2[image_index][x, j]
                # print(f" ({j},{y}): {pixel_value},")
                y_at_x_array.append(pixel_value)
            y_at_x_outer.append(y_at_x_array)

        x_array1 = list(range(0, 302))

        # plot x_at_50
        darkened_color = (0.2, 0.2, 0.2)
        transparency = 0.6
        plt.plot(x_array[0:50],x_at_y_outer[0][0:50],color = darkened_color, label = 'Sample 1',alpha = transparency)
        #plt.xlabel("X coordinates")
        #plt.ylabel("Pixel Values")
        #plt.show()
        plt.plot(x_array, x_at_y_outer[1],color = darkened_color,label = 'Sample 2',alpha = transparency)
        plt.plot(x_array, x_at_y_outer[2],color = darkened_color,label = 'Sample 3',alpha = transparency)

        plt.plot(x_array, x_at_y_outer[3],color = darkened_color,label = 'Sample 4',alpha = transparency)

        plt.plot(x_array, x_at_y_outer[4],color = darkened_color, label='Sample 5',alpha = transparency)

        plt.plot(x_array, x_at_y_outer[5],color = darkened_color, label='Sample 6',alpha = transparency)


        plt.xlabel("X coordinates")
        plt.ylabel("Pixel Values")
        plt.legend()
        plt.show()

        print(x_array1, "\n")
        print(y_at_x_outer[0], "\n")
        print(y_at_x_outer[1])

        yarn_count_2 = 0
        for i in range(0, 302):
            count = 0
            for j in range(0, 6):
                #print(y_at_x_outer[j][i])
                if y_at_x_outer[j][i] == 255.0:
                    count = count + 1
            print(f"count at {i}:{count}")
            if count == 6:
                yarn_count_2 = yarn_count_2 + 1
        print(yarn_count_2)"""


        ###########################################################################################################################
        yarn_count1 = 21
        yarn_count2 = 21
        print(f"warps count:{yarn_count1}")
        print(f"wefts count:{yarn_count2}")

        #crimp percentage
        yarn_length_fabric_x = 5 #cm
        straighten_yarn_length_x = 6.2 #warp sobelx 6incre

        yarn_length_fabric_y = 5  # cm
        straighten_yarn_length_y = 6.2  # weft sobely 6incre


        mass_of_unit_length = 0.0002
        needed_x = 9.74
        needed_y = 9.74

        warp_length = cal_original_length(straighten_yarn_length_x,yarn_length_fabric_x,needed_x)
        #print(f"original length of a warp:{warp_length}")

        weft_length = cal_original_length(straighten_yarn_length_y,yarn_length_fabric_y, needed_y)
        #print(f"original length of a weft:{weft_length}")

        mass_warps = (yarn_count1 * warp_length * mass_of_unit_length)
        mass_wefts = (yarn_count2 * weft_length * mass_of_unit_length)

        total_mass_of_yarns = mass_warps + mass_wefts
        area = needed_x * needed_y
        GSM = (total_mass_of_yarns*10000)/area

        print(f"GSM of image captured:{GSM}")





if __name__ == "__main__":
    main()
