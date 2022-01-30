import numpy as np
import ut as utils

class Novelty:
  def __init__(self, knn):
    self.knn = knn
    self.archive = []
  
  def get_novelty_metric(self, arch1, arch2, verbose=False):
    normal1, reduce1 = arch1.normal, arch1.reduce
    normal2, reduce2 = arch2.normal, arch2.reduce
    index, common_ops = 0, 0
    while index < len(normal1):
      if normal1[index] in normal2[index:index+2]: common_ops += 1
      if normal1[index+1] in normal2[index:index+2]: common_ops += 1
      if reduce1[index] in reduce2[index:index+2]: common_ops += 1
      if reduce1[index+1] in reduce2[index:index+2]: common_ops += 1
      index += 2
      if verbose: print(index, common_ops)
    similarity_metric = common_ops / (len(normal1) + len(reduce1) + len(normal2) + len(normal2) - common_ops)
    distance_metric = 1 - similarity_metric

    return distance_metric, similarity_metric
  
  def how_novel(self, arch, arch_idx, current_pop):
    '''
      Output: (Novelty score, k-nearest neighbors)
    '''
    tmp_novelty_rec = []
    
    # Novelty w.r.t the current population
    for idx, ind in enumerate(current_pop):
      if idx==arch_idx: continue
      distance, similarity = self.get_novelty_metric(arch, ind)
      tmp = (ind, distance)
      tmp_novelty_rec.append(tmp)
    # Novelty w.r.t the archive
    for ind in self.archive:
      distance, similarity = self.get_novelty_metric(arch, ind)
      tmp = (ind, distance)
      tmp_novelty_rec.append(tmp)
    # k Nearest Neighbor
    sorted_ = tmp_novelty_rec.sort(key=lambda x:x[1])
    KNNeighbors = tmp_novelty_rec[:self.knn]
    novelty_score = sum([neighbor[1] for neighbor in KNNeighbors]) / len(KNNeighbors)
    
    return novelty_score, KNNeighbors      
  
  def update_archive(self, novel_archs):
    for arch in novel_archs:
      if not utils.search_genotype_list(arch=arch, arch_list=self.archive):
        self.archive.append(arch) 
