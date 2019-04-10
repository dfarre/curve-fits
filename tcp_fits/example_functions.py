class MaxPartitionDiffV1:
    @staticmethod
    def zero(array):
        return max([abs(max(array[:k+1]) - max(array[k+1:])) for k in range(len(array) - 1)])

    @staticmethod
    def one(array):
        return max(abs(max(array[:k+1]) - max(array[k+1:])) for k in range(len(array) - 1))

    @staticmethod
    def two(array):
        numbers = set(array)

        return max(map(lambda s: abs(max(s) - max(numbers - s)),
                       (set(array[:k+1]) for k in range(len(array) - 1))))


class MaxPartitionDiffV2:
    @staticmethod
    def zero(array):
        lmax, rmax = array[0], max(array[1:])
        output = abs(lmax - rmax)

        for k in range(1, len(array) - 1):
            change = False

            if array[k] == rmax:
                rmax = max(array[k+1:])
                change = True

            if array[k] > lmax:
                lmax = array[k]
                change = True

            if change:
                output = max({output, abs(lmax - rmax)})

        return output

    @staticmethod
    def one(array):
        lmax, rmax = array[0], max(array[1:])
        output = abs(lmax - rmax)

        for k in range(1, len(array) - 1):
            change = False

            if array[k] == rmax:
                rmax = max(array[k+1:])
                change = True

            if array[k] > lmax:
                lmax = array[k]
                change = True

            if change:
                new_output = abs(lmax - rmax)

                if new_output > output:
                    output = new_output

        return output
