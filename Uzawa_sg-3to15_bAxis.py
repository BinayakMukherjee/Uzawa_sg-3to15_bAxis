import numpy

def coeff_E1_E1(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + (h[i] ** 8)
    return coeff


def coeff_E1_E2(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 2 * (h[i] ** 4) * (k[i] ** 4)
    return coeff


def coeff_E1_E3(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 2 * (h[i] ** 4) * (l[i] ** 4)
    return coeff


def coeff_E1_E4(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 4 * (h[i] ** 6) * (l[i] ** 2)
    return coeff


def coeff_E1_E5(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 4 * (h[i] ** 4) * (k[i] ** 2) * (l[i] ** 2)
    return coeff


def coeff_E1_E6(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 4 * (h[i] ** 6) * (k[i] ** 2)
    return coeff


def coeff_E1_E7(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 8 * (h[i] ** 7) * (l[i])
    return coeff


def coeff_E1_E8(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 8 * (h[i] ** 5) * (l[i] ** 3)
    return coeff


def coeff_E1_E9(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 8 * (h[i] ** 5) * (l[i]) * (k[i] ** 2)
    return coeff
# ----------------------------------#
def coeff_E2_E2(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + (k[i] ** 8)
    return coeff


def coeff_E2_E3(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 2 * (k[i] ** 4) * (l[i] ** 4)
    return coeff


def coeff_E2_E4(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 4 * (h[i] ** 2) * (k[i] ** 4) * (l[i] ** 2)
    return coeff


def coeff_E2_E5(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 4 * (k[i] ** 6) * (l[i] ** 2)
    return coeff


def coeff_E2_E6(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 4 * (h[i] ** 2) * (k[i] ** 6)
    return coeff


def coeff_E2_E7(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 8 * (h[i] ** 3) * (k[i] ** 4) * (l[i])
    return coeff


def coeff_E2_E8(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 8 * (h[i]) * (k[i] ** 4) * (l[i] ** 3)
    return coeff


def coeff_E2_E9(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 8 * (h[i]) * (k[i] ** 6) * (l[i])
    return coeff
# -----------------------------------#
def coeff_E3_E3(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + (l[i] ** 8)
    return coeff


def coeff_E3_E4(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 4 * (h[i] ** 2) * (k[i] ** 2) * (l[i] ** 4)
    return coeff


def coeff_E3_E5(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 4 * (k[i] ** 2) * (l[i] ** 6)
    return coeff


def coeff_E3_E6(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 4 * (h[i] ** 2) * (l[i] ** 6)
    return coeff


def coeff_E3_E7(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 8 * (h[i] ** 3) * (k[i]) * (l[i] ** 4)
    return coeff


def coeff_E3_E8(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 8 * (h[i]) * (k[i] ** 3) * (l[i] ** 4)
    return coeff


def coeff_E3_E9(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 8 * (h[i]) * (k[i]) * (l[i] ** 6)
    return coeff
# ----------------------------------#
def coeff_E4_E4(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 4 * (h[i] ** 4) * (k[i] ** 4)
    return coeff


def coeff_E4_E5(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 8 * (h[i] ** 2) * (k[i] ** 4) * (l[i] ** 2)
    return coeff


def coeff_E4_E6(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 8 * (h[i] ** 4) * (k[i] ** 2) * (l[i] ** 2)
    return coeff


def coeff_E4_E7(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 16 * (h[i] ** 5) * (k[i] ** 3)
    return coeff


def coeff_E4_E8(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 16 * (h[i] ** 3) * (k[i] ** 5)
    return coeff


def coeff_E4_E9(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 16 * (h[i] ** 3) * (k[i] ** 3) * (l[i] ** 2)
    return coeff
# ----------------------------------#
def coeff_E5_E5(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 4 * (k[i] ** 4) * (l[i] ** 4)
    return coeff


def coeff_E5_E6(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 8 * (h[i] ** 2) * (k[i] ** 2) * (l[i] ** 4)
    return coeff


def coeff_E5_E7(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 16 * (h[i] ** 3) * (k[i] ** 3) * (l[i] ** 2)
    return coeff


def coeff_E5_E8(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 16 * (h[i]) * (k[i] ** 5) * (l[i] ** 2)
    return coeff


def coeff_E5_E9(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 16 * (h[i]) * (k[i] ** 3) * (l[i] ** 4)
    return coeff
# ----------------------------------#
def coeff_E6_E6(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 4 * (h[i] ** 4) * (l[i] ** 4)
    return coeff


def coeff_E6_E7(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 16 * (h[i] ** 5) * (k[i]) * (l[i] ** 2)
    return coeff


def coeff_E6_E8(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 16 * (h[i] ** 3) * (k[i] ** 3) * (l[i] ** 2)
    return coeff


def coeff_E6_E9(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 16 * (h[i] ** 3) * (k[i]) * (l[i] ** 4)
    return coeff
# ----------------------------------#
def coeff_E7_E7(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 16 * (h[i] ** 6) * (k[i] ** 2)
    return coeff


def coeff_E7_E8(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 32 * (h[i] ** 4) * (k[i] ** 4)
    return coeff


def coeff_E7_E9(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 32 * (h[i] ** 4) * (k[i] ** 4)
    return coeff
# ----------------------------------#
def coeff_E8_E8(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 16 * (h[i] ** 2) * (k[i] ** 6)
    return coeff


def coeff_E8_E9(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 32 * (h[i] ** 2) * (k[i] ** 4) * (l[i] ** 2)
    return coeff
# ----------------------------------#
def coeff_E9_E9(h, k, l):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff + 16 * (h[i] ** 2) * (k[i] ** 2) * (l[i] ** 4)
    return coeff

# ----------------------------------#

def coeff_E1(h, k, l, gamma_expt):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff - 2 * (gamma_expt[i]) * (h[i] ** 4)
    return coeff


def coeff_E2(h, k, l, gamma_expt):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff - 2 * (gamma_expt[i]) * (k[i] ** 4)
    return coeff


def coeff_E3(h, k, l, gamma_expt):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff - 2 * (gamma_expt[i]) * (l[i] ** 4)
    return coeff


def coeff_E4(h, k, l, gamma_expt):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff - 4 * (gamma_expt[i]) * (h[i] ** 2) * (k[i] ** 2)
    return coeff


def coeff_E5(h, k, l, gamma_expt):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff - 4 * (gamma_expt[i]) * (k[i] ** 2) * (l[i] ** 2)
    return coeff


def coeff_E6(h, k, l, gamma_expt):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff - 4 * (gamma_expt[i]) * (h[i] ** 2) * (l[i] ** 2)
    return coeff


def coeff_E7(h, k, l, gamma_expt):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff - 8 * (gamma_expt[i]) * (h[i] ** 3) * (k[i])
    return coeff


def coeff_E8(h, k, l, gamma_expt):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff - 8 * (gamma_expt[i]) * (h[i]) * (k[i] ** 3)
    return coeff


def coeff_E9(h, k, l, gamma_expt):
    coeff = 0
    for i in range(0, len(h)):
        coeff = coeff - 8 * (gamma_expt[i]) * (k[i]) * (h[i]) * (l[i] ** 2)
    return coeff

# ----------------------------------#
def gamma(h,k,l,E):
    gamma_list =[]
    for i in range(0, len(h)):
        gamma_list.append((h[i]**4)*E[0] + (k[i]**4)*E[1] + (l[i]**4)*E[2] + 2*(h[i]**2)*(l[i]**2)*E[3] + 2*(l[i]**2)*(k[i]**2)*E[4] + 2*(h[i]**2)*(k[i]**2)*E[5] + 4*(h[i]**3)*(l[i])*E[6] + 4*(h[i])*(l[i]**3)*E[7] +\
        4*(h[i])*(l[i])*(k[i]**2)*E[8])
    return numpy.asarray(gamma_list)
# ----------------------------------#
def C_elements(h, k, l):
    return [h ** 4, k ** 4, l ** 4, 2 * (h ** 2) * (l ** 2), 2 * (l ** 2) * (k ** 2), 2 * (h ** 2) * (k ** 2),
            4 * (h ** 3) * (l), 4 * (l ** 3) * (h), 4 * h * l * (k ** 2)]

# ----------------------------------#
def read_integers(filename):
    numbers = open(filename).readlines()
    list = []
    for i in range(0, len(numbers)):
        list.append(float(numbers[i]))
    return list


# ----------------------------------#
def iterate(A, b, C, seed_u, seed_lambda):
    lambda0 = seed_lambda
    #print(lambda0)
    u0 = seed_u
    # print(A)
    #print(u0)
    iteration = 0
    for iteration in range (0,100000,1):
        lambda1 = numpy.maximum(lambda0 + 0.8 * (C @ u0), numpy.zeros(23))
        #print("lambda 1 ", lambda1)
        # print(b.transpose()-(numpy.matrix.transpose(C) @ lambda1))
        # print(numpy.linalg.inv(A))
        u1 = numpy.linalg.inv(A) @ (b.transpose() - (C.transpose() @ lambda1).transpose()).transpose()
        # print ("u1", u1)
        # print("convergence = {}\n".format(u1 - u0))
        lambda0 = lambda1.transpose()
        diff = abs(u1.transpose()[0] - u0)

        if numpy.all(diff < 0.00000001):
            break
        u0 = u1.transpose()[0]

        #print("lambda 0 new", lambda0)
        # print("u0 new ", u0)

        iteration += 1
    print("No. of iterations =", iteration)
    print("The new E's are:\n", u1)
    print("Diff between final and penultimate E's:", diff)
    return u1


# ----------------------------------#

def main():
    hkl = open("hkl.txt").readlines()
    h = []
    k = []
    l = []
    for i in range(0, len(hkl)):
        row = (hkl[i].split(" "))
        h.append(int(row[0]))
        k.append(int(row[1]))
        l.append(int(row[2]))
    E_expt_list = (read_integers('E_expt.txt'))
    E_expt = numpy.asarray(E_expt_list)
    gamma_expt = [-279.03,-135.39,-55.25,-400.64,-187.92,-1412.60,-1227.65,-1169.31,-1483.39,-594.79,-2174.02,
    -1858.33,-195.07,-4464.52,-4411.03,-2166.28,-3064.74,-1349.88,-1040.29,-6349.12,-3599.96,-1863.09,-3006.65] #gamma(h,k,l,E_expt) #
    print("Initial gamma's:", gamma_expt)
    # print(h)
    # print(k)
    # print(l)
    A = numpy.array([[(coeff_E1_E1(h, k, l)), (coeff_E1_E2(h, k, l)), (coeff_E1_E3(h, k, l)), (coeff_E1_E4(h, k, l)),
                      (coeff_E1_E5(h, k, l)), (coeff_E1_E6(h, k, l)), (coeff_E1_E7(h, k, l)), (coeff_E1_E8(h, k, l)),
                      (coeff_E1_E9(h, k, l))],
                     [(coeff_E1_E2(h, k, l)), (coeff_E2_E2(h, k, l)), (coeff_E2_E3(h, k, l)), (coeff_E2_E4(h, k, l)),
                      (coeff_E2_E5(h, k, l)), (coeff_E2_E6(h, k, l)), (coeff_E2_E7(h, k, l)), (coeff_E2_E8(h, k, l)),
                      (coeff_E2_E9(h, k, l))],
                     [(coeff_E1_E3(h, k, l)), (coeff_E2_E3(h, k, l)), (coeff_E3_E3(h, k, l)), (coeff_E3_E4(h, k, l)),
                      (coeff_E3_E5(h, k, l)), (coeff_E3_E6(h, k, l)), (coeff_E3_E7(h, k, l)), (coeff_E3_E8(h, k, l)),
                      (coeff_E3_E9(h, k, l))],
                     [(coeff_E1_E4(h, k, l)), (coeff_E2_E4(h, k, l)), (coeff_E3_E4(h, k, l)), (coeff_E4_E4(h, k, l)),
                      (coeff_E4_E5(h, k, l)), (coeff_E4_E6(h, k, l)), (coeff_E4_E7(h, k, l)), (coeff_E4_E8(h, k, l)),
                      (coeff_E4_E9(h, k, l))],
                     [(coeff_E1_E5(h, k, l)), (coeff_E2_E5(h, k, l)), (coeff_E3_E5(h, k, l)), (coeff_E4_E5(h, k, l)),
                      (coeff_E5_E5(h, k, l)), (coeff_E5_E6(h, k, l)), (coeff_E5_E7(h, k, l)), (coeff_E5_E8(h, k, l)),
                      (coeff_E5_E9(h, k, l))],
                     [(coeff_E1_E6(h, k, l)), (coeff_E2_E6(h, k, l)), (coeff_E3_E6(h, k, l)), (coeff_E4_E6(h, k, l)),
                      (coeff_E5_E6(h, k, l)), (coeff_E6_E6(h, k, l)), (coeff_E6_E7(h, k, l)), (coeff_E6_E8(h, k, l)),
                      (coeff_E6_E9(h, k, l))],
                     [(coeff_E1_E7(h, k, l)), (coeff_E2_E7(h, k, l)), (coeff_E3_E7(h, k, l)), (coeff_E4_E7(h, k, l)),
                      (coeff_E5_E7(h, k, l)), (coeff_E6_E7(h, k, l)), (coeff_E7_E7(h, k, l)), (coeff_E7_E8(h, k, l)),
                      (coeff_E7_E9(h, k, l))],
                     [(coeff_E1_E8(h, k, l)), (coeff_E2_E8(h, k, l)), (coeff_E3_E8(h, k, l)), (coeff_E4_E8(h, k, l)),
                      (coeff_E5_E8(h, k, l)), (coeff_E6_E8(h, k, l)), (coeff_E7_E8(h, k, l)), (coeff_E8_E8(h, k, l)),
                      (coeff_E8_E9(h, k, l))],
                     [(coeff_E1_E9(h, k, l)), (coeff_E2_E9(h, k, l)), (coeff_E3_E9(h, k, l)), (coeff_E4_E9(h, k, l)),
                      (coeff_E5_E9(h, k, l)), (coeff_E6_E9(h, k, l)), (coeff_E7_E9(h, k, l)), (coeff_E8_E9(h, k, l)),
                      (coeff_E9_E9(h, k, l))]])
    # print(numpy.linalg.inv(A))

    b = numpy.array([[coeff_E1(h, k, l, gamma_expt)],
                     [coeff_E2(h, k, l, gamma_expt)],
                     [coeff_E3(h, k, l, gamma_expt)],
                     [coeff_E4(h, k, l, gamma_expt)],
                     [coeff_E5(h, k, l, gamma_expt)],
                     [coeff_E6(h, k, l, gamma_expt)],
                     [coeff_E7(h, k, l, gamma_expt)],
                     [coeff_E8(h, k, l, gamma_expt)],
                     [coeff_E9(h, k, l, gamma_expt)]])
    # print(b)
    C_list = []
    for i in range(0, len(h)):
        C_list.append(C_elements(h[i], k[i], l[i]))
    C = numpy.asarray(C_list) * (-1)
    # print(numpy.matrix.transpose(C))
    # print(C)
    seed_u = (read_integers('seed_u.txt'))
    # print (seed_u)
    seed_lambda = (read_integers('seed_lambda.txt'))
    # print(seed_lambda)
    E_fin = iterate(A, b, C, seed_u, seed_lambda)
    # print(gamma(h,k,l,E_expt))
    print("Optimized gamma's:\n", gamma(h,k,l,E_fin))
    return


main()