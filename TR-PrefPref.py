import numpy as np
import time
import multiprocessing


def PA(vertexes, probabilities, m):
  return np.random.choice(vertexes, m, replace=False, p=probabilities)



def init(m, N, gamma):  
  g = [[] for _ in range(m + 1 + N)]
  degrees = np.zeros(m + N + 1, dtype=np.int64)
  # degsums = np.zeros(m + N + 1, dtype=np.int64)
  deglaw = np.zeros(m + N + 1, dtype=np.int64)
  k = np.zeros(m + N + 1, dtype=np.int64)

  degree_sum = m * (m + 1)
  degrees[:m+1] = m
  maxdeg = m

  for i in range(m + 1):
    for j in range(m + 1):
      if i == j:
        continue
      g[i].append(j)
  return g, degrees, degree_sum, maxdeg, deglaw, k, m + 1


def generate(m, N, vertex, fun, iters, gamma, model):
  g, degrees, degree_sum, maxdeg, deglaw, k, cur = init(m, N, gamma)

  k = []
  p = model
  for iter_num in range(N):
    probs = degrees / degree_sum
    

    pA = PA(cur, probs[:cur], m)
    pAcur = 1
    vc = pA[0]  
    dVC = degrees[vc]    
    vc_degree_sum = 0.0
    vc_degrees = []
    connected = [vc]
    for i in g[vc]:
      vc_degree_sum += degrees[i]
      vc_degrees.append(degrees[i])
    vc_probs = vc_degrees / vc_degree_sum
    rA = PA(g[vc], vc_probs, dVC)
    rAcur = 0
    degrees[vc]+=1
    g[vc].append(cur)
    g[cur].append(vc)


    for i in range(1, m):
      if np.random.random() > p:
        while True:
          v = pA[pAcur]
          pAcur += 1
          if not (v in connected):
            break
      else:
        while True:
          if rAcur < dVC:
            v = rA[rAcur]
            rAcur += 1
          else:
            v = pA[pAcur]
            pAcur += 1
          if not (v in connected):
            break
      connected.append(v)
      degrees[v]+=1
      if degrees[v] > maxdeg:
        maxdeg = degrees[v]
      g[v].append(cur)
      g[cur].append(v)

    degrees[cur] += m
    if degrees[cur] > maxdeg:
      maxdeg = degrees[cur]
    degree_sum += 2 * m
    

    # cur += 1
      

      ##############################################################
      ##############################################################
      ##############################################################


    # for v in PA(cur, probs[:cur], m):
    #   degrees[v]+=1
    #   # degree_sum += 2 * m
    #   # degdeg[v] = degrees[v] ** gamma
    #   # degree_sum += degdeg[v]
    #   g[v].append(cur)
    #   g[cur].append(v)

    
    # # для избранных вершин степень, макс. степень, и т.д.
    # if (cur in iters):
    #   for v in vertex:
    #     if v <= cur:
    #       k += [cur, v, degrees[v], maxdeg, degrees[v]/ (maxdeg + 0.0), degree_sum]


    # суммы степеней соседей
    # подсчет Г
    if (cur in iters):
      gAmma = 0.0
      for i in range(cur + 1):
        degsums = 0
        for v in g[i]:
          degsums += degrees[v]
        gAmma += (degrees[i] ** 2) / degsums

      # вывод Г
      k += [cur, gAmma / (cur)]
  
    cur += 1


  # рапспределение степеней
  for i in range(m + N + 1):
    deglaw[degrees[i]] += 1
      
  return [k, deglaw]


def run_thread(m, N, gamma, fun, model, T, vertex, iters, i, ret_dict):
  k = []
  for p in range(T):
    print(p)
    a = generate(m, N, vertex, fun, iters, gamma, model)
    k += [a]
  
  ret_dict[i] = k
  

def run(m, N, gamma, T, model, fun, vertex, iters, num_workers=2):
  
  assert T % num_workers == 0

  procs = []
  return_dict = multiprocessing.Manager().dict()


  for i in range(num_workers):
    p = multiprocessing.Process(target=run_thread, args=(m, N, gamma, fun, model, T//num_workers, vertex, iters, i, return_dict))
    procs.append(p)
    p.start()

  for proc in procs:
    proc.join()

  m = []
  for it in return_dict.values():
    m += it

  return m


def calc_s_list(ver, g, degrees, degree_sum, **kwargs):
  a = 0
  for v in g[ver]:
    a += degrees[v]
  return [ver, degrees[ver], a, degree_sum]


print("6")

# Пример эксперимента
if __name__ == '__main__':
    # calc_time()

    multiprocessing.freeze_support()
    N = 10001
    T = 6
    M = 5   
    # p = 1.06
    gamma = 0.5
    vertex = [2, 5, 10, 50, 100]
    # iters = [10, 500, 1000, 5000, 10000, 20000, 40000, 80000, 100000, 120000, 140000, 160000, 180000, 200000, 220000, 240000, 260000]
    iters = [10, 100, 250, 500, 750, 1000, 2000, 3000, 4000, 5000, 10000, 20000, 40000, 80000, 100000, 120000, 140000, 160000, 180000, 200000, 220000, 240000, 260000]
    
    

    for M in [2,5,10,25,50, 100 ]:
      for p in [0.25, 0.50, 0.75 , 1]:
        res_experiment = run(M, N, gamma, T, p, calc_s_list, vertex, iters, num_workers=6)
    

    
        with open(f"TriadPrefPrefG-p{p}_degsP_m{M}_g{gamma}_{N}_{T}.txt", 'w') as f3:
          with open(f"TriadPrefPrefG-p{p}_degs_m{M}_g{gamma}_{N}_{T}.txt", 'w') as f4:
            for el in res_experiment:
              for el1 in el[0]:
                f3.write(str(el1))
                f3.write(" ")
              for el1 in el[1]:
                f4.write(str(el1))
                f4.write(" ")
              f3.write("\n")
              f4.write("\n")
    print("Done")

