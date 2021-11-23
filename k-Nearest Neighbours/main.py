import numpy as np
from math import dist

def createDataset(fileName):
  """this function transforms a csv into a dataset list.
  this method has a string parameter that contains the filename of your csv file."""
  return np.genfromtxt(fileName, delimiter=';', usecols=[1, 2, 3, 4, 5, 6, 7],
                       converters={5: lambda s: 0 if s == b"-1" else float(s),
                                   7: lambda s: 0 if s == b"-1" else float(s)})

def createDates(fileName):
  """this function transforms a csv into a list with dates.
  this method has a string parameter that contains the filename of your csv file."""
  return np.genfromtxt(fileName, delimiter=';', usecols=[0])

def labelData(dates):
  """this function labels the given datasets with the correct season.
  this method has a integer list parameter that contains the data that you want to label."""
  labels = []
  for label in dates:
    if (label < 20000301) or (20010000 < label < 20010301):
      labels.append('winter')
    elif (20000301 <= label < 20000601) or (20010301 <= label < 20010601):
      labels.append('lente')
    elif (20000601 <= label < 20000901) or (20010601 <= label < 20010901):
      labels.append('zomer')
    elif (20000901 <= label < 20001201) or (20010901 <= label < 20011201):
      labels.append('herfst')
    else: # from 01-12 to end of year
      labels.append('winter')
  return labels

def calculateBestK(data, dataLabels, testData, testDataLabels, k_range_min = 1, k_range_max = 100):
  """this function calculates the optimal K by testing all the k's in range
  and checking the accuracy by comparing the predicted label with the actual label.
  :return: optimal K value"""
  best_k_accuracy = 0
  best_k_index = 0

  for k in range(k_range_min, k_range_max):
    correct_estimate = 0

    for objectIndex in range(len(testData)):
      if(kNN(data, dataLabels, testData[objectIndex], k) == testDataLabels[objectIndex]):
        correct_estimate += 1

    k_accuracy = (correct_estimate / len(testData)) * 100
    print(k, ' is the current K with an accuracy of: ', k_accuracy, '%')
    if k_accuracy > best_k_accuracy:
      best_k_accuracy = k_accuracy
      best_k_index = k

  print(best_k_index, ' is the most accurate K with an accuracy of: ', best_k_accuracy, '%')
  return best_k_index

# def getKey(dct, value):
#   """this function returns all the keys that match the specified value.
#   :param dct: the dict you're trying to lookup
#   :param value: the value you're trying to lookup in the dict
#   :return: the values that match the value_key
#   """
#   return [key for key in dct if (dct[key] == value)]

def kNN(data, dataLabels, testData, k):
  """this function calculates the distances with a euclidean distance formula between datasetPoints and then appends them to a list.
  after having appended all distances to a list, it will sort the list from smallest distance to largest and will then
  put k amount of distances in a new list called nearestNeighbours. this list will be used to count the most occuring label.
  after counting all the labels the function will check if there is a tie. If it sees a tie it will remove a neighbour
  from the list until there are no more ties. after which it will return the predicted label.
  :return: the predicted label by calculating the most occuring label"""
  distances = []
  nearestNeighbours = [0]*k

  for objectIndex in range(len(data)):
    distances.append([dist(testData, data[objectIndex]), objectIndex]) # het berekenen van de euclidian distance kan
                      # met verschillende library functies gedaan worden maar ik heb voor math.dist(a, b) gekozen,
                      # omdat het de snelste euclidean distance functie was en er geen verschil is in accuraatheid.
                      # bron: https://www.delftstack.com/howto/numpy/calculate-euclidean-distance/
                      # bij mij was de distance.euclidian(a, b) functie alleen wel het traagst met een average time to finish van 39.4 seconden
                      # dist(a, b) average time to finish 12.5 seconden
                      # temp = a-b, np.sqrt(np.dot(temp.T, temp)) average time to finish 16.4 seconden
                      # np.linalg.norm(a-b) average time to finish 24.7 seconden
                      # np.sqrt(np.sum(np.square(a-b))) average time to finish 29.5 seconden

  distances.sort()

  for kIndex in range(k):
    nearestNeighbours[kIndex] = distances[kIndex][1]

  mostOccuringItem = []
  tie = True

  while tie:
    tie = False
    count = {}

    for neighbour in nearestNeighbours:
      labelString = dataLabels[neighbour]
      if labelString in count:
        count[labelString] += 1
      else:
        count[labelString] = 1

    mostOccuringItem = max(count.items(), key= lambda x: x[1])

    for labelString in count:
      if (count[labelString] == mostOccuringItem[1]) and (labelString != mostOccuringItem[0]):
        # for neighbour in nearestNeighbours: # method that sorts by taking the nearest k neighbour that tied
        #   if dataLabels[neighbour] in getKey(count, mostOccuringItem[1]):
        #     return dataLabels[neighbour]

        nearestNeighbours.pop(len(nearestNeighbours) - 1)
        tie = True
        break

  return mostOccuringItem[0]

def main():
  data = createDataset('dataset1.csv')
  labels = labelData(createDates('dataset1.csv'))

  validation_Data = createDataset('validation1.csv')
  validation_Labels = labelData(createDates('validation1.csv'))

  unlabeled_Data = createDataset('days.csv')
  estimated_labels = []

  optimal_K = calculateBestK(data, labels, validation_Data, validation_Labels)

  for items in unlabeled_Data:
    estimated_labels.append(kNN(data, labels, items, optimal_K))

  for items in estimated_labels:
    print(items)

if __name__ == "__main__":
  import time
  start_time = time.time()
  main()
  print("--- %s seconds ---" % (time.time() - start_time))