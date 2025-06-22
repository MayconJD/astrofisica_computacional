import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import tkinter as tk
from tkinter import ttk, messagebox
from scipy.spatial import KDTree

CSV_FILE = 'sdss_data.csv'
H0 = 70.0
c = 299792.458

def load_and_preprocess_data(csv_path):
    try:
        df = pd.read_csv(csv_path, skiprows=1, low_memory=False)
        df.columns = df.columns.str.lower()

        required_cols = ['ra', 'dec', 'redshift', 'u', 'g', 'r', 'i', 'z']
        if not all(col in df.columns for col in required_cols):
            messagebox.showerror("Erro de Dados", f"Colunas essenciais {required_cols} não encontradas no CSV. Verifique o cabeçalho.")
            raise ValueError("Colunas ausentes no CSV.")

        df.dropna(subset=['redshift', 'u', 'g', 'r', 'i', 'z'], inplace=True)
        df = df[df['redshift'] > 0]

        df['distance_mpc'] = df['redshift'] * c / H0

        ra_rad = np.deg2rad(df['ra'])
        dec_rad = np.deg2rad(df['dec'])
        df['x'] = df['distance_mpc'] * np.cos(dec_rad) * np.cos(ra_rad)
        df['y'] = df['distance_mpc'] * np.cos(dec_rad) * np.sin(ra_rad)
        df['z_cartesian'] = df['distance_mpc'] * np.sin(dec_rad)

        df['g_r_color'] = df['g'] - df['r']
        df['u_g_color'] = df['u'] - df['g']
        df['r_i_color'] = df['r'] - df['i']

        df['M_r'] = df['r'] - (5 * np.log10(df['distance_mpc'].replace(0, np.nan)) + 25)
        df.dropna(subset=['M_r'], inplace=True)

        print("Dados carregados e pré-processados com sucesso.")
        return df

    except FileNotFoundError:
        messagebox.showerror("Erro de Arquivo", f"Arquivo '{csv_path}' não encontrado. Certifique-se de que está na mesma pasta do script.")
        exit()
    except Exception as e:
        messagebox.showerror("Erro de Processamento", f"Ocorreu um erro ao carregar/processar os dados: {e}")
        exit()

full_df = load_and_preprocess_data(CSV_FILE)
if full_df is None or full_df.empty:
    messagebox.showerror("Erro de Dados", "O DataFrame está vazio após o pré-processamento. Verifique seus dados.")
    exit()

min_redshift = full_df['redshift'].min()
max_redshift = full_df['redshift'].max()
redshift_range_diff = (max_redshift - min_redshift) / 5
redshift_options = []
for i in range(5):
    start_z = min_redshift + i * redshift_range_diff
    end_z = min_redshift + (i + 1) * redshift_range_diff
    if i == 4:
        end_z = max_redshift
    redshift_options.append(f"{start_z:.3f} - {end_z:.3f}")

filtered_df_global = pd.DataFrame()

def plot_galaxy_positions(filtered_df):
    if filtered_df.empty:
        messagebox.showinfo("Sem Dados", "Nenhuma galáxia encontrada no intervalo de redshift selecionado para plotar.")
        return

    fig = px.scatter_3d(filtered_df, x='x', y='y', z='z_cartesian',
                        color='g_r_color',
                        color_continuous_scale=px.colors.sequential.Plasma,
                        size_max=2, opacity=0.7,
                        title=f'Distribuição 3D de Galáxias (Redshift: {current_redshift_range.get()})',
                        labels={'x': 'X (Mpc)', 'y': 'Y (Mpc)', 'z_cartesian': 'Z (Mpc)', 'g_r_color': 'Cor (g-r)'})
    fig.update_traces(marker=dict(size=1))
    
    output_html_file = "galaxy_positions_3d.html"
    fig.write_html(output_html_file)
    messagebox.showinfo("Gráfico Salvo", f"O gráfico 3D foi salvo como '{output_html_file}'. Abra-o em seu navegador.")

def plot_redshift_dimension_distribution(filtered_df):
    if filtered_df.empty:
        messagebox.showinfo("Sem Dados", "Nenhuma galáxia encontrada no intervalo de redshift selecionado para plotar.")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(filtered_df['redshift'], filtered_df['r'], s=5, alpha=0.5, c=filtered_df['g_r_color'], cmap='viridis')
    plt.xlabel('Redshift')
    plt.ylabel('Magnitude Aparente (r-band)')
    plt.title(f'Magnitude Aparente vs. Redshift para Galáxias (Redshift: {current_redshift_range.get()})')
    plt.colorbar(label='Cor (g-r)')
    plt.grid(True)
    plt.show()

def plot_color_magnitude_diagram(filtered_df):
    if filtered_df.empty:
        messagebox.showinfo("Sem Dados", "Nenhuma galáxia encontrada no intervalo de redshift selecionado para plotar.")
        return
    
    if 'M_r' not in filtered_df.columns:
        messagebox.showerror("Erro de Coluna", "Coluna 'M_r' não encontrada. Verifique o pré-processamento dos dados.")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(filtered_df['g_r_color'], filtered_df['M_r'], s=5, alpha=0.5, c=filtered_df['redshift'], cmap='plasma')
    plt.xlabel('Cor (g-r)')
    plt.ylabel('Magnitude Absoluta (M_r)')
    plt.title(f'Diagrama Cor-Magnitude (Redshift: {current_redshift_range.get()})')
    plt.gca().invert_yaxis()
    plt.colorbar(label='Redshift')
    plt.grid(True)
    plt.show()

def update_info():
    selected_range_str = current_redshift_range.get()
    if not selected_range_str:
        info_label.config(text="Selecione um intervalo de redshift.")
        return

    start_z_str, end_z_str = selected_range_str.split(' - ')
    start_z = float(start_z_str)
    end_z = float(end_z_str)

    global filtered_df_global
    filtered_df_global = full_df[(full_df['redshift'] >= start_z) & (full_df['redshift'] < end_z)].copy()

    num_galaxies = len(filtered_df_global)
    
    if num_galaxies > 0:
        avg_r_mag = filtered_df_global['r'].mean()
        avg_g_r_color = filtered_df_global['g_r_color'].mean()
        avg_distance = filtered_df_global['distance_mpc'].mean()

        info_text = (
            f"Intervalo de Redshift: {selected_range_str}\n"
            f"Número de Galáxias: {num_galaxies}\n"
            f"Magnitude 'r' Média: {avg_r_mag:.2f}\n"
            f"Cor 'g-r' Média: {avg_g_r_color:.2f}\n"
            f"Distância Média: {avg_distance:.2f} Mpc"
        )
    else:
        info_text = (
            f"Intervalo de Redshift: {selected_range_str}\n"
            f"Número de Galáxias: 0\n"
            f"Nenhuma galáxia encontrada neste intervalo."
        )
    
    info_label.config(text=info_text)

def on_plot_positions_click():
    update_info()
    plot_galaxy_positions(filtered_df_global)

def on_plot_redshift_distribution_click():
    update_info()
    plot_redshift_dimension_distribution(filtered_df_global)

def on_plot_color_magnitude_click():
    update_info()
    plot_color_magnitude_diagram(filtered_df_global)

root = tk.Tk()
root.title("Explorador de Galáxias por Redshift")

main_frame = ttk.Frame(root, padding="10")
main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

ttk.Label(main_frame, text="Selecione o Intervalo de Redshift:").grid(row=0, column=0, sticky=tk.W, pady=5)
current_redshift_range = tk.StringVar(root)
redshift_combobox = ttk.Combobox(main_frame, textvariable=current_redshift_range, values=redshift_options, state="readonly")
redshift_combobox.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
redshift_combobox.set(redshift_options[0])
redshift_combobox.bind("<<ComboboxSelected>>", lambda event: update_info())

info_label = ttk.Label(main_frame, text="Selecione um intervalo para ver as informações.", justify=tk.LEFT)
info_label.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

button_frame = ttk.Frame(main_frame)
button_frame.grid(row=3, column=0, columnspan=2, pady=10)

ttk.Button(button_frame, text="1. Plotar Gráfico de Posição (3D)", command=on_plot_positions_click).grid(row=0, column=0, padx=5, pady=5)
ttk.Button(button_frame, text="2. Gráfico de Mag. vs. Redshift", command=on_plot_redshift_distribution_click).grid(row=0, column=1, padx=5, pady=5)
ttk.Button(button_frame, text="3. Diagrama Cor-Magnitude", command=on_plot_color_magnitude_click).grid(row=0, column=2, padx=5, pady=5)

update_info()

root.mainloop()