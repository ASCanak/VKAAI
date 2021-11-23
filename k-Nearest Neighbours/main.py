from scipy.spatial import distance
import numpy as np

def labelData(dates): #labels the datasets with the correct season
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
  best_k_accuracy = 0
  best_k_index = 0

  for k in range(k_range_min, k_range_max):
    correct_estimate = 0

    for i in range(len(testData)):
      if(kNN(data, dataLabels, testData[i], k) == testDataLabels[i]):
        correct_estimate += 1

    k_accuracy = (correct_estimate / len(testData)) * 100
    print(k, ' is the current K with an accuracy of: ', k_accuracy, '%')
    if k_accuracy > best_k_accuracy:
      best_k_accuracy = k_accuracy
      best_k_index = k

  print(best_k_index, ' is the most accurate K with an accuracy of: ', best_k_accuracy, '%')
  return best_k_index

def kNN(data, dataLabels, testData, k):
  distances = []
  nearestNeighbours = [0]*k

  for i in range(len(data)):
    distances.append([distance.euclidean(testData, data[i]), i])

  distances.sort()

  for i in range(k):
    nearestNeighbours[i] = distances[i][1]

  countDict = dict()
  mostOccuringString = ""
  mostOccuringNumber = 0
  for neighbour in nearestNeighbours:
    labelString = dataLabels[neighbour]
    if labelString in countDict.keys():
      countDict[labelString] += 1
    else:
      countDict[labelString] = 1
    if countDict[labelString] > mostOccuringNumber:
      mostOccuringString = labelString
      mostOccuringNumber = countDict[labelString]
  return mostOccuringString

def main():
  data = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[1, 2, 3, 4, 5, 6, 7],
                       converters={5: lambda s: 0 if s == b"-1" else float(s),
                                   7: lambda s: 0 if s == b"-1" else float(s)})
  dates = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[0])
  labels = labelData(dates)

  data_validation = np.genfromtxt('validation1.csv', delimiter=';', usecols=[1, 2, 3, 4, 5, 6, 7],
                                  converters={5: lambda s: 0 if s == b"-1" else float(s),
                                              7: lambda s: 0 if s == b"-1" else float(s)})
  dates_validation = np.genfromtxt('validation1.csv', delimiter=';', usecols=[0])
  labels_validation = labelData(dates_validation)

  unlabeled_Data = np.genfromtxt('days.csv', delimiter=';', usecols=[1, 2, 3, 4, 5, 6, 7],
                                  converters={5: lambda s: 0 if s == b"-1" else float(s),
                                              7: lambda s: 0 if s == b"-1" else float(s)})

  optimal_K = calculateBestK(data, labels, data_validation, labels_validation)

  estimated_labels = []
  for items in unlabeled_Data:
    estimated_labels.append(kNN(data, labels, items, optimal_K))

  for items in estimated_labels:
    print(items)

if __name__ == "__main__":
  main()