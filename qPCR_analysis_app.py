"""
==================
qPCR Analysis Tool
==================
Version 1.0
------------------
This script performs:
- Calculates: Mean Cq, ΔCq, ΔΔCq with corresponding SED (Based on Livak Mehtod)
- Supports:
    - Single reference sample
    - Multiple treatment/control groups (normalizing to more than one sample)
-log2 transformation
-Plot generation (SVG)
- Excel raw data export
-----------------------
Plotting is shown on a log2 scale (Meaning a value of 3 means 2^3 fold change up-regulation)
=========================
Usage:  Open your Pycharm Terminal and type: shiny run --reload app.py
        Then click on the link which is shown in the terminal (e.g. Uvicorn running on http://127.0.0.1:8000)
        Now you can do everything in the application, jsut let pycharm run in the background
        If somehting is not working, you can check the terminal of pycharm for error codes
        Also in case some of your wells were omitted and not included in the analysis, it will be displayed in the terminal as well
---------------------------
Author: Lukas Senftleben
"""

from shiny import App, render, ui, reactive
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


matplotlib.rcParams['svg.fonttype'] = 'none'
import numpy as np
import io

# Terminal command to  run the web app and have it reloading after every change in the script: shiny run --reload app.py
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.markdown("### 1. Data Upload"),
        ui.input_file("file_input", "Select qPCR .csv", accept=[".csv"]),
        ui.input_select("task_type", "Task Type", ["Standard", "Unknown"]),
        ui.busy_indicators.options(spinner_type="dots", spinner_color="#007bff"),
        ui.hr(),
        ui.markdown("### 2. Gene Settings"),
        ui.input_select("hk_gene", "Housekeeping Gene", choices= ["GAPDH"]),
        ui.input_select("backup_gene", "Backup Gene (to ignore)", choices=["TBP"]),
        ui.hr(),
        ui.markdown("### 3. Normalization Settings"),
        ui.input_checkbox("use_one_ref", "Use single Reference Sample", value=True),
        ui.panel_conditional(
            "input.use_one_ref",
            ui.input_select("ref_sample", "Single Ref Sample Name", choices=[]),
        ),
        ui.br(),
        ui.input_checkbox("use_multi_ref", "Use Treated/Control Groups", value=True),
        ui.panel_conditional("input.use_multi_ref",
        ui.input_text("treated_kw", "Treated Keyword", value="DOX"),
            ui.input_text("control_kw", "Control Keyword", value="w/o DOX"),
                             ),

        ui.hr(),
        ui.markdown("### 4. Plot Appearance"),
        #------Gene filtering----
        ui.input_checkbox("plot_all_genes", "Plot all target genes", value=True),
        ui.panel_conditional( "!input.plot_all_genes", ui.input_checkbox_group("plot_selected_genes", "Select Genes to Plot", choices=[])),
        ui.input_select("plot_style", "Background Style", ["whitegrid", "darkgrid", "ticks"]),
        ui.input_select("palette", "Color Palette", ["colorblind", "muted", "viridis", "Set2", "bright", "pastel"]),
        ui.input_checkbox("plot_dcq_check", "Include Delta Cq Plot", value=False),
        ui.hr(),
        ui.markdown("#### Granular Font Sizes (pt)"),
        ui.input_checkbox("show_title", "Show Plot Title", value=False),
        ui.input_numeric("size_title", "Plot Title", value=11, min=4, max=40),
        ui.input_numeric("size_axes_labels", "Axis Labels (X/Y)", value=10, min=4, max=40),
        ui.input_numeric("size_ticks", "Tick Numbers", value=8, min=4, max=40),
        ui.input_numeric("size_legend_title", "Legend Header", value=9, min=4, max=40),
        ui.input_numeric("size_legend_text", "Legend Genes", value=9, min=4, max=40),
        ui.input_checkbox("rotation", "Rotate X axis description", value=False),
        ui.input_numeric("fig_width_cm", "Figure width (cm)", value=16, min=5),
        ui.input_numeric("fig_height_cm", "Figure height (cm)", value=10, min=4),
        ui.markdown("""<small><b> Tips for Sizes: </b><br>
            - Journal (1-col): 9 x 7 cm<br>
            - PPT Slide: 16 x 10 cm<br>
            - Poster: 30 x 18 cm</small>"""),
        ui.hr(),
        ui.input_text("filename", "SVG File Name:", value="my_plot"),
        ui.input_action_button("run", "Analyze & Generate Plots", class_="btn-primary w-100"),
        ui.br(),
        ui.download_button("download_report", "Download Raw Data (.xlsx)", class_="btn-success w-100"),
        title="Settings"  # This adds a title to the sidebar itself
    ),
    # The "Main Panel" is now just the rest of the page inside page_sidebar
    ui.panel_title("qPCR Analysis Tool"),
# --- MAIN CONTENT TABS ---
    ui.navset_card_pill(
        ui.nav_panel(
            "Data Preview",
            # Corrected: style moved inside a ui.div
            ui.div(
                ui.output_table("data_preview"),
                style="overflow-x: auto; padding: 10px;"
            )
        ),

        ui.nav_panel(
            "Delta Cq",
            ui.panel_conditional(
                "input.plot_dcq_check",
                # Added the scrollable div here as well for consistency
                ui.div(
                    ui.output_plot("plot_dcq", fill=True),
                    ui.download_button("download_dcq_svg", "Save Plot as SVG", class_="btn-sm"),
                    style="height: 700px; overflow: auto; border: 1px solid #ddd; padding: 10px;"
                ),
            ),
            ui.panel_conditional(
                "!input.plot_dcq_check",
                ui.markdown("Check 'Include Delta Cq' in sidebar to see this analysis.")
            )
        ),

        ui.nav_panel(
            "Single Reference",
            ui.panel_conditional(
                "input.use_one_ref",
                ui.div(
                    ui.output_plot("plot_one_ref"),
                    ui.download_button("download_one_ref_svg", "Save Plot as SVG", class_="btn-sm"),
                    style="height: 700px; overflow: auto; border: 1px solid #ddd; padding: 10px;"
                ),
            ),
            ui.panel_conditional(
                "!input.use_one_ref",
                ui.markdown("Check 'Use Single Reference' in sidebar to see this analysis.")
            )
        ),

        ui.nav_panel(
            "Treated vs Control",
            ui.panel_conditional(
                "input.use_multi_ref",
                ui.div(
                    ui.output_plot("plot_multi_ref"),
                    ui.download_button("download_multi_ref_svg", "Save Plot as SVG", class_="btn-sm"),
                    style="height: 700px; overflow: auto; border: 1px solid #ddd; padding: 10px;"
                ),
            ),
            ui.panel_conditional(
                "!input.use_multi_ref",
                ui.markdown("Check 'Use Treated/Control Groups' in sidebar to see this analysis.")
            )
        ),
    ))

# =============Server=========================
def server(input, output, session):
    #figsize helper for cm to inches
    def fig_size_inches(width_cm, height_cm):
        width_cm = max(5, min(width_cm, 50))
        height_cm = max(4, min(height_cm, 40))
        return width_cm / 2.54, height_cm / 2.54
    #--------------------------Reactive updater for choices based on uploaded CSV--------------------------
    @reactive.effect
    def _():
        file_info = input.file_input()
        if file_info:
            # Read the file just to get the sample names
            df = pd.read_csv(file_info[0]["datapath"], sep=",", on_bad_lines='skip')
            # Get unique samples, remove empty ones, and sort them
            samples = df['Sample'].dropna().unique().tolist()
            df['Target'] = df['Target'].astype(str).str.strip().str.upper()
            targets = df['Target'].dropna().unique().tolist()
            task_types = df['Task'].dropna().unique().tolist()
            # Update the sidebar dropdown
            ui.update_select("ref_sample", choices=samples)
            ui.update_select("hk_gene", choices=targets)
            ui.update_select("backup_gene", choices=targets)
            ui.update_select("task_type", choices=task_types)
            ui.update_checkbox_group("plot_selected_genes", choices=targets, selected=targets)
        # ========================== PLOTTING-Global Settings ===========================================
        # Helper function for data filtering before plotting
    def filter_plot_data(df):
        if input.plot_all_genes():
            return df
        selected = input.plot_selected_genes()
        return df[df['Gene'].isin(selected)]
        #plot function
    def make_barplot(df, x_col, y_label, title, hue='Gene', xlabel=None):
        sns.set_theme(style=input.plot_style(), palette=input.palette())
        df = filter_plot_data(df)
        fig, ax = plt.subplots(figsize=fig_size_inches(
            input.fig_width_cm(),
            input.fig_height_cm()))
        # Bars

        sns.barplot(
            x=x_col,
            y='log2_FC',
            hue=hue,
            data=df,
            ax=ax,
            errorbar=None
        )
        # Baseline
        ax.axhline(0, color='black', linestyle='--', linewidth=1.0)
        # ---------Error Bars--------------
        # size adaption
        current_area_delta = np.sqrt(input.fig_width_cm() * input.fig_height_cm())
        baseline_area_delta = np.sqrt(20.0 * 12.0)
        size_multiplier = current_area_delta / baseline_area_delta
        dynamic_capsize = max(1.5, 2.0 * size_multiplier)
        dynamic_linewidth = max(0.4, 1.0 * size_multiplier)
        for bar, (_, row) in zip(ax.patches, df.iterrows()):
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            sd = row.get('SD', 0)
            if not np.isnan(y):
                ax.errorbar(
                    x=x, y=y, yerr=sd,
                    fmt='none', ecolor='black',
                    capsize=dynamic_capsize,
                    linewidth=dynamic_linewidth
                )
        # Fontsizes
        if input.show_title():
            ax.set_title(title, fontsize=input.size_title())
        else:
            ax.set_title("")
        ax.set_ylabel(y_label, fontsize=input.size_axes_labels())
        ax.set_xlabel("")
        ax.tick_params(axis='both', labelsize=input.size_ticks())
        unique_labels = df[x_col].unique()
        if input.rotation():
            ax.set_xticks(range(len(unique_labels)))
            ax.set_xticklabels(
                unique_labels,
                rotation=45,
                ha='right',
                rotation_mode='anchor',
                fontsize=input.size_axes_labels()
            )
        else:
            ax.set_xticks(range(len(unique_labels)))
            ax.set_xticklabels(
                unique_labels,
                rotation=0,
                ha='center',
                fontsize=input.size_axes_labels()
            )
        # 4. Legend
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            title=hue,
            title_fontsize=input.size_legend_title(),
            fontsize=input.size_legend_text()
        )
        fig.tight_layout()
        return fig
    #==========================Math engine=================================
    @reactive.calc
    @reactive.event(input.run)
    def results():
        with ui.Progress(min=1, max=3) as p:
            p.set(message="Calculaiton in progress", detail="loading CSV...")
            file_info = input.file_input()
            if file_info is None:
                return None
            # --- Step 1: Trimming------
            df_raw = pd.read_csv(file_info[0]["datapath"], sep=",", encoding='utf-8', on_bad_lines='skip')
            df_raw_no_neg = df_raw[df_raw['Task'] == input.task_type()]
            clean_df = df_raw_no_neg[df_raw_no_neg['Omit'] == False].copy()
            essential_cols = ['Well', 'Omit', 'Sample', 'Target', 'Task', 'Cq', 'Cq Conf']
            clean_df = clean_df[essential_cols]
            clean_df['Target'] = clean_df['Target'].astype(str).str.strip().str.upper()
            clean_df['Sample'] = clean_df['Sample'].astype(str).str.strip()
            clean_df['Cq'] = pd.to_numeric(clean_df['Cq'], errors='coerce')
            #Memorize original order from CSV
            original_samples = clean_df['Sample'].unique().tolist()
            original_targets = clean_df['Target'].unique().tolist()

            # --- Step2: Pivot,Averages & SEM ---
            av_std = clean_df.groupby(['Sample', 'Target'])['Cq'].agg(['mean', 'std', 'count']).reset_index()
            pivot_df = av_std.pivot(index='Sample', columns='Target', values='mean')
            pivot_std = av_std.pivot(index='Sample', columns='Target', values='std').fillna(0)
            #Reorder Samples and Targets based on original CSV
            ordered_samples = [s for s in original_samples if s in pivot_df.index]
            pivot_df = pivot_df.reindex(index=ordered_samples)
            pivot_std = pivot_std.reindex(index=ordered_samples)
            ordered_targets = [t for t in original_targets if t in pivot_df.columns]
            pivot_df = pivot_df[ordered_targets]
            pivot_std = pivot_std[ordered_targets]
            # --- Step 3: Delta Cq ---
            hk = input.hk_gene().strip().upper()
            backup = input.backup_gene().strip().upper()
            if hk not in pivot_df.columns:
                print(f"Error: {hk} not found in CSV columns: {pivot_df.columns.tolist()}")
                return None
            target_genes = [g for g in pivot_df.columns if g not in [hk, backup]] #filtering out hk and backup, creating list only with target genes
            delta_cq_results = pd.DataFrame(index=pivot_df.index)
            delta_cq_std = pd.DataFrame(index=pivot_df.index)
            delta_cq_combined = pd.DataFrame(index=pivot_df.index)
            for gene in target_genes:
                val_mean = pivot_df[gene] - pivot_df[hk]
                val_std = np.sqrt(pivot_std[gene]**2 + pivot_std[hk]**2)
                #store in single tables for further calculations
                delta_cq_results[gene] = val_mean
                delta_cq_std[gene] = val_std
                #store in combined table(only for display)
                delta_cq_combined[f"{gene}_ΔCq"] = val_mean
                delta_cq_combined[f"{gene}_ΔCq_std"] = val_std
                #adding melted data for seaborn plotting
                df_log2_dcq = -delta_cq_results
                df_dcq_long = df_log2_dcq.reset_index().melt(id_vars='Sample', var_name='Gene', value_name='log2_FC')
                df_dcq_std_long = delta_cq_std.reset_index().melt(id_vars='Sample', var_name='Gene', value_name='SD')
                df_delta_cq_plot = df_dcq_long.merge(df_dcq_std_long, on=['Sample', 'Gene'])


            output_data = {
                "clean": clean_df,
                "averages": pivot_df,
                "delta_cq": delta_cq_combined,
                "log2_fc_dcq": -pivot_df[target_genes].sub(pivot_df[hk], axis=0),
                "dcq_plot_data": df_delta_cq_plot,
            }
            # --- Step 4: MULTI-REFERENCE ddCq ANALYSIS ---
            if input.use_multi_ref():

                t_kw = input.treated_kw().strip()
                c_kw = input.control_kw().strip()

                df = delta_cq_results.copy()
                df_std = delta_cq_std.copy()

                # 1. Parse structured name: Group__Condition__Role
                def parse(name):
                    try:
                        group, condition, role = name.split("__")
                    except ValueError:
                        raise ValueError(
                            f"Invalid sample name: {name}\n"
                            "Expected format: Group__Condition__Role"
                        )
                    return group, condition, role
                parsed = df.index.to_series().apply(parse)
                df[["Group", "Condition", "Role"]] = pd.DataFrame(parsed.tolist(), index=df.index)
                parsed_std = df_std.index.to_series().apply(parse)
                df_std[["Group", "Condition", "Role"]] = pd.DataFrame(parsed_std.tolist(), index=df_std.index)

                # 2. Apply UI-defined treated/control classification
                treated = df[df.index.str.contains(t_kw, regex=False)]
                control = df[df.index.str.contains(c_kw, regex=False)]
                treated_std = df_std[df_std.index.str.contains(t_kw, regex=False)]
                control_std = df_std[df_std.index.str.contains(c_kw, regex=False)]
                # 3. Match by Condition
                results = []
                results_std = []

                for idx, t_row in treated.iterrows():

                    condition = t_row["Condition"]

                    c_match = control[control["Condition"] == condition]

                    if c_match.empty:
                        raise ValueError(
                            f"No control sample found for condition '{condition}' "
                            f"for treated sample '{idx}'"
                        )

                    if len(c_match) > 1:
                        raise ValueError(
                            f"Multiple controls found for condition '{condition}'"
                        )

                    c_row = c_match.iloc[0]

                    t_std = treated_std.loc[idx, target_genes]
                    c_std = control_std.loc[c_match.index[0], target_genes]

                    ddcq = t_row[target_genes] - c_row[target_genes]

                    ddcq_std = np.sqrt(
                        t_std.astype(float) ** 2 +
                        c_std.astype(float) ** 2
                    )

                    ddcq.name = idx
                    ddcq_std.name = idx

                    results.append(ddcq)
                    results_std.append(ddcq_std)

                # --------------------------------------------------------
                # 4. Build outputs
                # --------------------------------------------------------
                df_ddCq = pd.DataFrame(results)
                df_ddCq_std = pd.DataFrame(results_std)
                df_ddCq.index.name = "Sample"
                df_ddCq_std.index.name = "Sample"
                df_log2 = -df_ddCq.copy()
                df_log2.index.name = "Sample"
                df_plot = (
                    df_log2.reset_index()
                    .rename(columns={"index": "Sample"})
                    .melt(id_vars="Sample", var_name="Gene", value_name="log2_FC")
                    .merge(
                        df_ddCq_std.reset_index()
                        .rename(columns={"index": "Sample"})
                        .melt(id_vars="Sample", var_name="Gene", value_name="SD"),
                        on=["Sample", "Gene"]
                    )
                )

                df_plot = df_plot.dropna(subset=["log2_FC", "SD"])

                # --------------------------------------------------------
                # 5. Output
                # --------------------------------------------------------
                output_data["multi_plot_data"] = df_plot
                output_data["multi_log2_fc"] = df_ddCq
                output_data["multi_std"] = df_ddCq_std
            # ========== Step 5: ONE REF SAMPLE ================
            if input.use_one_ref():
                ref = input.ref_sample()
                if ref in delta_cq_results.index:
                    # 1. Extract reference row and SD
                    ref_row = delta_cq_results.loc[ref]
                    ref_std = delta_cq_std.loc[ref]

                    # 2. Subtract reference from all samples
                    ddcq_df = delta_cq_results.sub(ref_row, axis=1)
                    ddcq_std = np.sqrt(delta_cq_std ** 2 + ref_std ** 2)

                    # 3. Log2 transformation
                    log2_fc = -ddcq_df

                    # 4. Transform to long format
                    index_name = log2_fc.index.name or 'Sample'
                    df_log2 = log2_fc.reset_index().melt(id_vars=index_name, var_name='Gene', value_name='log2_FC')
                    df_log2_err = ddcq_std.reset_index().melt(id_vars=index_name, var_name='Gene', value_name='SD')

                    # Merge values and errors
                    df_long_ref = df_log2.merge(df_log2_err, on=[index_name, 'Gene'])

                    # 5. Filter out the reference sample itself and NaNs
                    df_intermediate = df_long_ref[df_long_ref[index_name] != ref]
                    intended_genes = set(df_intermediate['Gene'].unique())

                    df_long_ref = df_long_ref[df_long_ref['log2_FC'].notna()]
                    df_long_ref = df_long_ref[df_long_ref[index_name] != ref]

                    df_single_plot = df_long_ref.copy()

                    failed_genes = intended_genes - set(df_single_plot['Gene'].unique())
                    if failed_genes:
                        print(f"Note: Gene(s) hidden due to NaN value(s) in Single Ref: {failed_genes}")

                    # Store the long-format plot data and the wide-format Excel data
                    output_data["single_plot_data"] = df_single_plot
                    output_data["single_log2_fc"] = -ddcq_df  # Kept for Excel export
                    output_data["single_std"] = ddcq_std
            #Step 6: Building Master tables for preview and xlsx export
            master_base = pd.DataFrame(index=pivot_df.index)
            for gene in target_genes:
                master_base[f"{gene}_Cq_Mean"] = pivot_df[gene]
                master_base[f"{gene}_Cq_SD"] = pivot_std[gene]
                master_base[f"{gene}_ΔCq"] = delta_cq_results[gene]
                master_base[f"{gene}_ΔCq_SD"] = delta_cq_std[gene]
                if input.use_one_ref() and "single_log2_fc" in output_data:
                    master_base[f"{gene}_ΔΔCq_Log2_FC"] = output_data["single_log2_fc"][gene]
                    master_base[f"{gene}_ΔΔCq_Log2_FC_SD"] = output_data["single_std"][gene]
            output_data["master_base"] = master_base

            if input.use_multi_ref() and "multi_log2_fc" in output_data:
                master_group =pd.DataFrame(index=output_data["multi_log2_fc"].index)
                for gene in target_genes:
                    master_group[f"{gene}_ΔΔCq_Log2FC"] = output_data["multi_log2_fc"][gene]
                    master_group[f"{gene}_ΔΔCq_Log2FC_SD"] = output_data["multi_std"][gene]
                output_data["master_group"] = master_group

            ui.notification_show("Analysis complete!", type="message", duration=5)
        return output_data


    # --------------------dcq Plot-------------------------
    def make_plot_dcq():
        res = results()
        if res is None or "dcq_plot_data" not in res:
            return None

        df = filter_plot_data(res["dcq_plot_data"])

        return make_barplot(
            df=df,
            x_col=df.columns[0],
            y_label=f"log2 Fold Change (ΔCq)",
            title="Delta Cq values (log2)",
        )
    @render.plot
    def plot_dcq():
        return make_plot_dcq()
    #-----------------------------ddcq Plot for one reference sample-------------------------------
    def make_plot_one_ref():
        res = results()
        if res is None or "single_plot_data" not in res:
            return None

        df = filter_plot_data(res["single_plot_data"])
        ref_name = input.ref_sample()

        return make_barplot(
            df=df,
            x_col=df.columns[0],
            y_label=f"log2 Fold Change (−ΔΔCq)",
            title="Relative Gene Expression (Log2 Scale)",
        )
    @render.plot
    def plot_one_ref():
        return make_plot_one_ref()
    #--------------------Treated VS Control Plotting--------------------

    def make_plot_multi_ref():
        res = results()
        if res is None or "multi_plot_data" not in res:
            return None

        df = filter_plot_data(res["multi_plot_data"])
        t_kw = input.treated_kw().strip()
        c_kw = input.control_kw().strip()

        return make_barplot(
            df=df,
            x_col='Sample',
            y_label=f"log2 Fold Change (−ΔΔCq)",
            title="Relative Gene Expression (Treated vs Control)",
        )
    @render.plot
    def plot_multi_ref():
        return make_plot_multi_ref()
    @render.table
    def data_preview():
        res = results()
        if res is None : return None
        return res["master_base"].reset_index().head(15)

    # ===========THE DOWNLOAD LOGIC ===============
    @render.download(filename="qPCR_raw_Data.xlsx") #Download table as xlsx
    def download_report():
        res = results()
        if res is None: return None
        with io.BytesIO() as buf:
            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                #Sheet 1
                res["clean"].to_excel(writer, sheet_name='Cleaned Raw Data')
                #Sheet 2
                res["master_base"].to_excel(writer, sheet_name='Cq Calculation')
                #Sheet3
                if "master_group" in res:
                    res["master_group"].to_excel(writer, sheet_name='ΔΔCq of Treated vs Control')
            buf.seek(0)
            yield buf.getvalue()
    #----------Download DeltaCq Plot--------------------
    @render.download(filename="qPCR_DeltaCq_Plot.svg")
    def download_dcq_svg():
        fig = make_plot_dcq()  # This calls your existing plot function
        if fig is None: return None
        with io.BytesIO() as buf:
            fig.savefig(buf, format="svg",)
            plt.close(fig)
            yield buf.getvalue()
    #-----------Download for Single Reference Plot----------------------
    @render.download(filename=lambda: f"{input.filename()}.svg")
    def download_one_ref_svg():
        fig = make_plot_one_ref()
        if fig is None: return None
        with io.BytesIO() as buf:
            fig.savefig(buf, format="svg")
            plt.close(fig)
            yield buf.getvalue()
    #-------------------Download for Treated vs Control Plot-------------------------
    @render.download(filename=lambda: f"{input.filename()}.svg")
    def download_multi_ref_svg():
        fig = make_plot_multi_ref()
        if fig is None: return None
        with io.BytesIO() as buf:
            fig.savefig(buf, format="svg")
            plt.close(fig)
            yield buf.getvalue()
#starting the app
app = App(app_ui, server)
