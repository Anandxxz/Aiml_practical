
import pandas as pd
import matplotlib.pyplot as plt


data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'Exam_Marks': [35, 40, 50, 55, 65, 70, 80, 90]
}

df = pd.DataFrame(data)

correlation = df['Study_Hours'].corr(df['Exam_Marks'])

print("Correlation Coefficient:", round(correlation, 2))


if correlation > 0:
    print("Positive Association")
elif correlation < 0:
    print("Negative Association")
else:
    print("No Linear Association")


plt.scatter(df['Study_Hours'], df['Exam_Marks'])
plt.xlabel("Study Hours")
plt.ylabel("Exam Marks")
plt.title("Study Hours vs Exam Marks")
plt.grid(True)
plt.show()
