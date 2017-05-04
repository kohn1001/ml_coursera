def cost(theta, p):
 
    y = 2*pow(theta, p) 
    y += 2
    return y


def calc(theta, epc):
    res = cost(theta+epc, 3)
    res2 = cost(theta-epc, 3)
    #kres =2
    epc2 = 2*epc
    return (res-res2)/epc2

print(calc(1, 0.01))
