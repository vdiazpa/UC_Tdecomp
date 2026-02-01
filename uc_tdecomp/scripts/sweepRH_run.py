#sweep.RH.py

# df = sweep_RH( data, T = T, F_vals = [12,24], L_vals = [12], seeds=(41, 86, 55), opt_gap = 0.05, only_valid = True, csv_path = f"rh_duke_results_EXP_{T}HR_sto.csv", verbose = True)
# T = 168
# data =  load_csv_data(T)
# df2 = sweep_RH( data, T = T, F_vals = [12,24], L_vals = [12], seeds=(41, 86, 55), opt_gap = 0.05, only_valid = True, csv_path = f"rh_duke_results_EXP_{T}HR_sto.csv", verbose = True)
# T = 336
# data =  load_csv_data(T)
# df3 = sweep_RH( data, T = T, F_vals = [12,24], L_vals = [12], seeds=(41, 86, 55), opt_gap = 0.05, only_valid = True, csv_path = f"rh_duke_results_EXP_{T}HR_sto.csv", verbose = True)




##############Plotting code##############

# import pandas as pd

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# csv_path = r"C:\Users\vdiazpa\Documents\Egret\rh_T36_seed1.csv"

# df = pd.read_csv(csv_path)

# x   = df["F"]
# y   = df["L"]
# ofv = df["avg_ofv"]   
# runtime = df["avg_time"]
# target = ofv

# fig = plt.figure()
# ax  = fig.add_subplot(111, projection='3d')

# ax.scatter(x, y, target, c=target, cmap='viridis', s=50)
# ax.set_xlim(0, 35)
# ax.set_ylim(0, 35)
# ax.view_init(elev = 25, azim=135)

# ax.set_xlabel('F')
# ax.set_ylabel('L')
# ax.set_zlabel('Average Runtime (s)')

# plt.title('Average Runtime by F and L')
# plt.show()

# # Runtime heatmap
# rt = df.pivot(index="L", columns="F", values="avg_time")

# fig1, ax1 = plt.subplots()
# im1 = ax1.imshow(rt, origin="lower", aspect="auto", cmap="viridis")
# ax1.set_xlabel("F"); ax1.set_ylabel("L")
# ax1.set_title("Runtime heatmap")
# fig1.colorbar(im1, ax=ax1, label="Avg runtime (s)")

# # OFV heatmap
# ofv = df.pivot(index="L", columns="F", values="avg_ofv")
# fig4, ax4 = plt.subplots()
# im4 = ax4.imshow(ofv, origin="lower", aspect="auto", cmap="coolwarm")
# ax4.set_xlabel("F"); ax4.set_ylabel("L")
# ax4.set_title("OFV heatmap")
# # show ticks aligned with F/L values
# ax4.set_xticks(np.arange(ofv.shape[1])); ax4.set_yticks(np.arange(ofv.shape[0]))
# ax4.set_xticklabels(ofv.columns, rotation=90); ax4.set_yticklabels(ofv.index)
# fig4.colorbar(im4, ax=ax4, label="Avg OFV")

# # show figures (necessary in scripts)
# plt.tight_layout()
# plt.show()


# # Scatter Runtime vs OFV
# fig2, ax2 = plt.subplots()
# sc = ax2.scatter(df["avg_time"], df["avg_ofv"], c=(df["F"] + df["L"]), s=25, alpha=0.7, cmap="plasma")
# ax2.set_xlabel("Avg runtime (s)"); ax2.set_ylabel("Avg OFV")
# ax2.set_title("Runtime vs OFV")
# fig2.colorbar(sc, ax=ax2, label="F+L (window size)")

# # add monolithic markers
# mon_runtime = 29.3   # seconds, at 0.05 optimality gap
# mon_runtime = 29.2   # seconds, at 0.1 optimality gap
# mon_ofv     = 487321336.34 

# # vertical line for monolithic runtime and horizontal line for monolithic OFV
# ax2.axvline(x=mon_runtime, color='black', linestyle='--', linewidth=1, label='Monolithic runtime, 0.05 opt gap')
# ax2.axhline(y=mon_ofv, color='black', linestyle='-.', linewidth=1, label='Monolithic OFV, 0.05 opt gap')

# # annotate the intersection (small boxed label with arrow)
# ax2.annotate(f"Monolithic\nruntime={mon_runtime:.1f}s\nOFV={mon_ofv:.2f}",
#              xy=(mon_runtime, mon_ofv), xycoords='data',
#              xytext=(10,10), textcoords='offset points',
#              fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
#              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"))

# ax2.legend(loc='best', fontsize=8)

# # optional: log-scaled runtime heatmap in its own figure
# fig3, ax3 = plt.subplots()
# im3 = ax3.imshow(np.log1p(rt), origin="lower", aspect="auto", cmap="magma")
# ax3.set_xlabel("F"); ax3.set_ylabel("L")
# ax3.set_title("Log Runtime heatmap")
# fig3.colorbar(im3, ax=ax3, label="log(1 + avg runtime)")

# show figures (necessary in scripts)
# plt.show()
