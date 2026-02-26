from Critical_Velocity_Methods import *

single_plot = True  # Set to True to plot all cases in a single figure

if single_plot == False:
    for i in range(0,7):
        name = str(i)

        path = os.path.join(project_path, name + "_ConstantAmplitude", "Flutter_Derivatives",name + "_")

        flutter_derivatives_points, flutter_derivatives_curve = Extract_Flutter_Derivatives(path)
        path_storage = os.path.join(project_path, name + "_ConstantAmplitude", "Results")
        os.makedirs(path_storage, exist_ok=True)
        Plot_Flutter_Derivatives(flutter_derivatives_points, flutter_derivatives_curve, path=path_storage)


if single_plot == True:

    def Plot_Flutter_Derivatives_Combined(all_data, path="_"):
        fig = plt.figure(figsize=(10, 12))
        gs = fig.add_gridspec(4, 2, wspace=0.3, hspace=0.4)
        U_reduced = np.arange(0.1, 25, 0.1)
        
        # Define colors for each dataset
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))
        
        for column, letter in enumerate("HA"):
            for row in range(4):
                deriv_ax = fig.add_subplot(gs[row, column])
                key = f"{letter}{row + 1}"
                
                # Plot each dataset
                for idx, data_dict in enumerate(all_data):
                    flutter_derivatives_points = data_dict['points']
                    flutter_derivatives_curve = data_dict['curve']
                    name = data_dict['name']
                    color = colors[idx]
                    
                    # Plot polynomial curve
                    polynomial_coef = flutter_derivatives_curve[key]
                    fitted_polynomial = polynomial_coef
                    
                    # Plot points (without labels)
                    if flutter_derivatives_points is not None:
                        data = flutter_derivatives_points[key]
                        deriv_ax.scatter(data[0], data[1], marker="+", color=color, alpha=0.6)
                        fitted_polynomial = np.polyval(polynomial_coef, U_reduced)
                    
                    # Plot lines with labels only (only on first subplot for legend)
                    deriv_ax.plot(U_reduced, fitted_polynomial, color=color, 
                                label=f"Amplitude {name}" if row == 0 and column == 0 else "")
                
                deriv_ax.set_ylabel(f"${letter}_{row + 1}$")
                deriv_ax.set_xlim([min(U_reduced), max(U_reduced)])
                
                # Add legend only to first subplot
                if row == 0 and column == 0:
                    deriv_ax.legend(loc="best", fontsize=6, ncol=2)
            
            deriv_ax.set_xlabel(r"$U_{red}$")
        
        fig.savefig(path + "/Flutter_Derivatives_Combined_Plot.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()


    def Plot_Individual_Flutter_Derivatives(all_data, path="_"):
        """
        Create separate plots for each flutter derivative (H1, H2, H3, H4, A1, A2, A3, A4)
        """
        U_reduced = np.arange(0.1, 25, 0.1)
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))
        
        # Create individual plots for each derivative
        derivatives = ["H1", "H2", "H3", "H4", "A1", "A2", "A3", "A4"]
        
        for key in derivatives:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot each dataset
            for idx, data_dict in enumerate(all_data):
                flutter_derivatives_points = data_dict['points']
                flutter_derivatives_curve = data_dict['curve']
                name = data_dict['name']
                color = colors[idx]
                
                # Plot polynomial curve
                polynomial_coef = flutter_derivatives_curve[key]
                fitted_polynomial = polynomial_coef
                
                # Plot points (without labels)
                if flutter_derivatives_points is not None:
                    data = flutter_derivatives_points[key]
                    ax.scatter(data[0], data[1], marker="+", color=color, alpha=0.6)
                    fitted_polynomial = np.polyval(polynomial_coef, U_reduced)
                
                # Plot lines with labels
                ax.plot(U_reduced, fitted_polynomial, color=color, label=f"Amplitude {name}")
            
            ax.set_xlabel(r"$U_{red}$", fontsize=12)
            ax.set_ylabel(f"${key[0]}_{key[1]}$", fontsize=12)
            ax.set_title(f"Flutter Derivative ${key[0]}_{key[1]}$", fontsize=14, fontweight='bold')
            ax.set_xlim([min(U_reduced), max(U_reduced)])
            ax.legend(loc="best", fontsize=10)
            ax.grid(True, alpha=0.3)
            
            fig.savefig(path + f"/Flutter_Derivative_{key}.png", dpi=300, bbox_inches="tight")
            plt.show()
            plt.close()

    # Collect all data first
    all_flutter_data = []
    for i in range(0, 7):
        name = str(i)
        path = os.path.join(project_path, name + "_ConstantAmplitude", "Flutter_Derivatives", name + "_")
        flutter_derivatives_points, flutter_derivatives_curve = Extract_Flutter_Derivatives(path)
        all_flutter_data.append({
            'name': name,
            'points': flutter_derivatives_points,
            'curve': flutter_derivatives_curve
        })

    # Save to combined results folder
    path_storage = os.path.join(project_path, "Combined_Results", "Flutter_Derivatives_Plot")
    os.makedirs(path_storage, exist_ok=True)

    # Plot all together
    Plot_Flutter_Derivatives_Combined(all_flutter_data, path=path_storage)

    # Plot individual flutter derivatives
    Plot_Individual_Flutter_Derivatives(all_flutter_data, path=path_storage)