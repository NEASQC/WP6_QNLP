from jinja2 import Template

def main():
  '''
  This is the only function in this script. It takes no arguments and
  returns nothing. The script reads a text file containing the models 
  and corresponding parameters that the user wishes to carry out 
  benchmarking experiments for. The format of this text file should 
  follow that specified in the subsequent comments within the script.
  The script then fills up a template with these parameters to produce 
  SLURM scripts. These can then be submitted to an HPC cluster to run
  the benchmark experiments.
  '''
  #we could add a prompt to the user to introduce path of the paramters text file
  params_input = open('benchmark_params.txt', 'r')
  j = 0
  lines = params_input.readlines()
  for line in lines:
      j += 1
      params = line.split()
      if params[0] == 'pre_alpha':
        if len(params) != 5: 
            print("WARNING: line " + str(j) + " in your text file has the wrong number of parameters for the chosen model. This line will be skipped.")
            continue
        else:  
          '''
          If model is pre-alpha, create a SLURM script whose name is in the format  
          pre_alpha_[seed]__[optimiser]_[iterations]_[runs].sh
          '''  
          slurm_template = "slurm_templates/pre_alpha.sh"
          with open(slurm_template, 'r') as f1:
            template = Template(f1.read())
          s, p, i, r = params[1:]
          filled_template = template.render(s=s, r=r, i=i, p=p)
          with open(f'slurm_scripts/pre_alpha_{s}_{p}_{i}_{r}.sh','w') as f2:
            f2.write(filled_template)
          
      elif params[0] == 'pre_alpha_lambeq':
        if len(params) != 10: 
          print("WARNING: line " + str(j) + " in your text file has the wrong number of parameters for the chosen model. This line will be skipped.")
          continue
        else:
          '''
          If model is pre-alpha lambeq, create a SLURM script whose name is in the format
          pre_alpha_lambeq_[seed]_[optimiser]_[iterations]_[runs]_[ansatz]_[qubits per noun]_[number of circuit layers]_[number of single qubit params]_[batch size].sh
          '''
          slurm_template = "slurm_templates/pre_alpha_lambeq.sh"
          with open(slurm_template, 'r') as f1:
            template = Template(f1.read())
          s, p, i, r, an, qn, nl, np, b = params[1:]
          filled_template = template.render(s=s, r=r, i=i, p=p, an=an, qn=qn,nl=nl,np=np, b=b)
          with open(f'slurm_scripts/pre_alpha_lambeq_{s}_{p}_{i}_{r}_{an}_{qn}_{nl}_{np}_{b}.sh','w') as f2:
            f2.write(filled_template)
    
      #elif params[0] = 'alpha'
      #here we will do the same with alpha if we ever decide we want to run benchmarking experiments with it

      elif params[0] == 'beta_neighbours':
        if len(params) != 2: 
          print("WARNING: line " + str(j) + " in your text file has the wrong number of parameters for the chosen model. This line will be skipped.")
          continue
        else:
          '''
          If model is beta_neighbours, create a SLURM script whose name is in the format
          beta_neighbours_[number of K neighbours].sh
          '''
          slurm_template = "slurm_templates/beta_neighbours.sh"
          with open(slurm_template, 'r') as f1:
            template = Template(f1.read())
          k = params[1]
          filled_template = template.render(k=k)
          with open(f'slurm_scripts/beta_neighbours_{k}.sh','w') as f2:
            f2.write(filled_template)

      elif params[0] == 'alpha_pennylane':
        if len(params) != 11: 
          print("WARNING: line " + str(j) + " in your text file has the wrong number of parameters for the chosen model. This line will be skipped.")
          continue
        else:
          '''
          If model is alpha pennylane, create a SLURM script whose name is in the format
          alpha_pennylane[seed]_[runs]_[iterations]_[n_qubits]_[q_delta]_[batch_size]_[learning_rate]_[weight_decay]_[step_lr]_[gamma].sh
          '''
          slurm_template = "slurm_templates/alpha_pennylane.sh"
          with open(slurm_template, 'r') as f1:
            template = Template(f1.read())
          s, r, i, nq, qd, sb, lr, wd, slr, g = params[1:]
          filled_template = template.render(s=s, r=r, i=i, nq=nq, qd=qd, sb=sb, lr=lr, wd=wd, slr=slr, g=g)
          with open(f'slurm_scripts/alpha_pennylane{s}_{r}_{i}_{nq}_{qd}_{sb}_{lr}_{wd}_{slr}_{g}.sh','w') as f2:
            f2.write(filled_template)

      elif params[0] == 'alpha_lambeq':
        if len(params) != 16: 
          print("WARNING: line " + str(j) + " in your text file has the wrong number of parameters for the chosen model. This line will be skipped.")
          continue
        else:
          '''
          If model is alpha lambeq, create a SLURM script whose name is in the format
          alpha_lambeq[seed]_[runs]_[iterations]_[version]_[pca]_[ansatz]_[qn]_[qs]_[nl]_[np]_[batch_size]_[learning_rate]_[weight_decay]_[step_lr]_[gamma].sh
          '''
          slurm_template = "slurm_templates/alpha_lambeq.sh"
          with open(slurm_template, 'r') as f1:
            template = Template(f1.read())
          s, r, i, v, pca, an, qn, qs, nl, np, sb, lr, wd, slr, g = params[1:]
          filled_template = template.render(s=s, r=r, i=i, v=v, pca=pca, an=an, qn=qn, qs=qs, nl=nl, np=np, sb=sb, lr=lr, wd=wd, slr=slr, g=g)
          with open(f'slurm_scripts/alpha_lambeq{s}_{r}_{i}_{v}_{pca}_{an}_{qn}_{qs}_{nl}_{np}_{sb}_{lr}_{wd}_{slr}_{g}.sh','w') as f2:
            f2.write(filled_template)


      else: print("WARNING: The model " + params[0] + " specified in line " + str(j) + " is not a valid model. Only pre_alpha, pre_alpha_lambeq, alpha_pennylane, alpha_lambeq and beta_neighbours are currently supported. This line will be skipped.")

  params_input.close()

if __name__ == "__main__":
  main()

