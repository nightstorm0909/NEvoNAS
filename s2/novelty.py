class Novelty:
  def __init__(self, knn):
    self.knn = knn
#     self.novelty_threshold = novelty_threshold
    self.archive = []
    
  def get_arch_from_str(self, arch_str, verbose=False):
    if verbose: print(f'Input architecture: {arch_str}')
    arch = []
    for node_ops in arch_str.split('+'):
      for element in node_ops.split('|'):
        if not (element==''): arch.append(element)
    assert len(arch)==6, 'Wrong number of nodes and operations'
    return arch

  def get_novelty_metric(self, arch1_str, arch2_str, verbose=False):
    # Positional <Something>
    arch1 = self.get_arch_from_str(arch1_str)
    arch2 = self.get_arch_from_str(arch2_str)
    if verbose: print(f'arch1: {arch1}, \narch2: {arch2}')

    common_ops = 0
    for idx in range(len(arch1)):
      if arch1[idx] == arch2[idx]: common_ops += 1

    similarity_metric = common_ops / (len(arch1) + len(arch2) - common_ops)
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


  def get_local_fitness_with_novelty(self, arch, arch_top1, arch_idx, current_pop, acc_df):
    novelty_score, KNNeighbors = self.how_novel(arch=arch,
                                                arch_idx=arch_idx,
                                                current_pop=current_pop)    
    local_fitness = 0
    for (ind, _) in KNNeighbors:
      series = acc_df[ acc_df['genotype']==ind ]
      assert not series.empty
      fitness = series['top1'].values[0]
      #print(fitness, ind)
      if arch_top1 > fitness:
        local_fitness += 1    
    return novelty_score, KNNeighbors, local_fitness
  
  def update_archive(self, novel_archs):
    for arch in novel_archs:
      if not(arch in self.archive):
        self.archive.append(arch)
    #print('here', self.archive)
  
