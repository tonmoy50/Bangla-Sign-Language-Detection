import math 
  
# Function to check  
# palindrome  
def isPalindrome(s): 
    left = 0
    right = len(s) - 1
    while (left <= right): 
        if (s[left] != s[right]): 
            return False
          
        left = left + 1
        right = right - 1
  
    return True
  
# Function to calculate  
# the sum of n-digit  
# palindrome  
def getSum(n): 
    start = 100  
    end = 990
  
    sum = 0
  
    # Run a loop to check  
    # all possible palindrome  
    for i in range(start, end + 1): 
        s = str(i)  
          
        # If palndrome  
        # append sum  
        if (isPalindrome(s)): 
            sum = sum + i 
  
    return sum
  
# Driver code  
  
n = 1
ans = getSum(n) 
print(ans) 