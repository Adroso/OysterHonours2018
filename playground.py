list = [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]

for position, value in enumerate(list): #assuming pixel 1 will never have an edge

    if position != len(list)-1:
        if value > 0 and list[position+1] == 0:
            starting_edge = position

        if value > 0 and list[position-1] ==0:
            ending_edge = position

            distance = ending_edge - starting_edge - 1 #take one to correct for list index
            #print(distance)


 inner_oyster = []
    for position, pixel_value in enumerate(vp):

        if pixel_value == 0:
            starting_edge = position
            pixel_value = 1
        else:
            starting_edge = position

        if pixel_value > 0 and vp[position-1] ==0:
            ending_edge = position
            pixel_distance = ending_edge - starting_edge
            if pixel_distance > 5:
                print(pixel_distance)
                major_distance_count += 1
                pixel_value = 1
            else:
                minor_distance_count += 1
                pixel_value = 1

            starting_edge = None


#copied 20/8
if pixel_value == 0:
    # starting_edge = position
    pixel_value = 1

    if vp[position - 1] > 0:
        starting_edge = position
        pixel_value = 0

    elif position + 1 < len(vp) and vp[position + 1] > 0:
        ending_edge = position
        pixel_distance = ending_edge - starting_edge
        if pixel_distance > 0:
            print(pixel_distance)
            major_distance_count += 1
            # pixel_value = 1
        else:
            minor_distance_count += 1
            # pixel_value = 1
        starting_edge = None