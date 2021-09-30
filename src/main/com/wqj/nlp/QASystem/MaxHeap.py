class MaxHeap(object):
    def __init__(self, arr: list):
        self.arr = arr
        self.mark = 1
        while self.mark == 1:
            self.build()

    def build(self):
        self.mark = 0  # 先置为零， 只要经过一次swap函数，就再次置为1
        index = len(self.arr) - 1
        for i in range(index):
            if i * 2 + 2 <= index:  # 如果左右两个子节点都存在，去比较他们的大小
                self.tri(i, i * 2 + 1, i * 2 + 2)
            elif i * 2 + 1 <= index:  # 如果只有左子节点存在，去比较他们的大小
                if self.arr[i] < self.arr[i * 2 + 1]:
                    self.swap(i, i * 2 + 1)
            else:
                break

    def tri(self, head: int, left: int, right: int):
        if self.arr[head] < self.arr[left]:
            self.swap(head, left)
        if self.arr[head] < self.arr[right]:
            self.swap(head, right)

    def swap(self, index_1: int, index_2: int):
        self.mark = 1
        temp = self.arr[index_2]
        self.arr[index_2] = self.arr[index_1]
        self.arr[index_1] = temp

    def show(self):
        print(self.arr)

    def pop(self) -> int:
        self.arr[0] = self.arr[-1]
        temp = self.arr.pop()
        self.mark = 1
        while self.mark == 1:
            self.build()
        return temp

    def push(self, value: int):
        self.arr.append(value)
        self.mark = 1
        while self.mark == 1:
            self.build()
