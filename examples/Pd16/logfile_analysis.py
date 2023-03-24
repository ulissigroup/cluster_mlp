
filename="clus_Pd16.log"

parent_calls_list = []
with open(filename, 'r') as fh:
    for line in fh:
        if line.startswith("Parent"):
            line = line.strip()
            words = line.split(':')
            parent_calls_list.append(words[1])
print(parent_calls_list)
print(len(parent_calls_list))
