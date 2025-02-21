import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Patch3D,Poly3DCollection
from mpl_toolkits.mplot3d import proj3d


def normalize_transparency(value, df, min_val=None, max_val=None):
    # Default min and max if not passed
    if min_val is None: min_val = df.min().min()
    if max_val is None: max_val = df.max().max()
    # Inverse scaling (smaller values should have higher transparency)
    alpha = 1-(max_val - value) / (max_val - min_val) * 0.7  # Transparency between 0.1 and 1
    #print(alpha)
    return f'rgba(0, 0, 0, {alpha})'  # Black color with varying alpha

# Step 3: Apply transparency to the DataFrame
def apply_transparency(val, df):
    # Generate rgba color based on the value in the DataFrame
    return f'color: {normalize_transparency(val, df)}'  # Apply transparency to text color

def style_table(s, df, precision = True, transparency = False):
    s.set_table_styles([
        # Column header style (lighter gray background for column headers)
        {
            'selector': 'thead th',
            'props': [
                ('background-color', '#F2F2F2'),  # Lighter gray background for columns
                ('color', 'black'),
                ('font-weight', 'bold'),
                ('border', '2px solid black'),
                ('font-family', 'Arial'),
                ('text-align', 'center'),
                ('vertical-align', 'middle'),
                ('font-size', '20px')  # Larger font size for headers
            ]
        },
        # Body row styling (for both odd/even rows)
        {
            'selector': 'tbody tr',
            'props': [
                ('background-color', 'white'),
                ('border', '2px solid black'),
                ('height', '60px'),
                ('width', '150px'),
                ('text-align', 'center'),  # Horizontally center the text
                ('vertical-align', 'middle'),  # Vertically center the text
                ('font-size', '20px'),  # Larger text
                ('padding', '5px'),  # Adequate padding for better appearance
                ('font-weight', 'bold')  # Bold table content
            ]
        },
        # Table cell styling (set vertical borders between cells)
        {
            'selector': 'tbody td',
            'props': [
                ('border-left', '2px solid black'),
                ('border-right', '2px solid black'),
                ('padding', '5px'),
                ('text-align', 'center'),
                ('vertical-align', 'middle'),
                ('font-size', '20px'),  # Larger font size for numbers in table body
                ('font-weight', 'normal')  # Bold content inside table cells
            ]
        },
        {
            'selector': 'thead th:first-child',
            'props': [
                ('background-color', 'transparent'),
                ('border', 'none')
            ]
        },
        # Index style (lighter gray background for index)
        {
            'selector': '.index_name',  # Index labels at the top
            'props': [
                ('background-color', '#F2F2F2'),  # Lighter gray background for index
                ('font-weight', 'bold'),  # Regular (not bold) for index labels
                ('font-family', 'Arial'),
                ('text-align', 'center'),  # Horizontally center the text
                ('vertical-align', 'middle'),  # Vertically center the text
            ]
        },
        # Index rows in the body (lighter gray background for index)
        {
            'selector': '.row_heading',  # Index labels in body rows
            'props': [
                ('background-color', '#F2F2F2'),  # Lighter gray background for index
                ('font-weight', 'bold')  # Regular (not bold) for index labels in body
            ]
        }
    ])
    # Format numeric values to 2 decimal places
    if precision:
        s.format(precision=2)

    if transparency:
        s.applymap(lambda x: apply_transparency(x, df))

    return s


def plot_cube(ax, alphas = [0.2]*6):
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face
    ])
    
    # Define the faces of the cube using lists of vertex indices
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left
        [vertices[1], vertices[2], vertices[6], vertices[5]]   # Right
    ]
    face_colors = [
    (0, 1, 1, alphas[0]),  # Cyan
    (1, 1, 0, alphas[1]), # Yellow
    (1, 0, 1, alphas[2]), # Magenta
    (1, 0, 0, alphas[3]),  # Red
    (0, 1, 0, alphas[4]),  # Green
    (0, 0, 1, alphas[5]),  # Blue

]

    
    # Create a Poly3DCollection object
    cube = Poly3DCollection(faces, 
                            facecolors=face_colors, 
                            linewidths=1, 
                            edgecolors='grey', 
                            linestyle = '--')
                            #alpha=0.2)
    
    # Add the collection to the plot
    ax.add_collection3d(cube)
    ax.view_init(elev=25, azim=20)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    # Get rid of the spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    
    # Get rid of the ticks
    ax.set_xticks([])#[0,1], [0,1]) 
    ax.set_yticks([])#[0,1], [0,1]) 
    ax.set_zticks([])#[0,1], [0,1])
    # Set axis labels

def plot_square(ax, color = 'blue'):
    square = patches.Rectangle((0., 0.), 1, 1,
                               facecolor = color,
                               linewidth=1, 
                            edgecolor='grey', 
                            linestyle = '--',
                            alpha=0.2)

    # Add the square to the plot
    ax.add_patch(square)
    
    # Set the aspect ratio to 'equal' to ensure the square's proportions are correct
    ax.set_aspect('equal')
    

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Remove the ticks
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none')

    
    # Get rid of the ticks
    ax.set_xticks([])#[0,1], [0,1]) 
    ax.set_yticks([])#[0,1], [0,1]) 
    # Set axis labels

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

def plot_data(xs, ys, zs, ax, lw = 2):
    flag = 1
    loop = False
    for j in range(len(xs)-1):
        x1, x2 = xs[j], xs[j+1]
        y1, y2 = ys[j], ys[j+1]
        z1, z2 = zs[j], zs[j+1]

        a = Arrow3D([x1, x2], [y1, y2], 
                    [z1, z2], mutation_scale=20,
                    #connectionstyle="arc3,rad=0.1",
                    lw=lw, arrowstyle="-|>", color="black",
                   zorder = 7)
        ax.add_artist(a)
        if x1 == x2 and y1 == y2 and z1 == z2:
                        # Parameters for the ellipse
            if loop:
                flag+=0.5
            a = 0.2*flag  # semi-major axis
            b = 0.07*flag  # semi-minor axis
            loop = True
            normal_vector = np.array([1,0,0
                                     ]) # Arbitrary normal vector for the ellipse
            point = np.array([x1, y1, z1])  # The pre-specified point through which the ellipse should pass
            
            # Plot the elliptical loop and get the final points for the arrow
            x_end, x_prev = plot_elliptical_loop(ax, 
                                                 a=a, b=b, 
                                                 lw = lw,
                                                 alpha = 1,
                                                 normal_vector=normal_vector, 
                                                 point=point)
        else:
            loop = False
            # Add an arrow at the end of the loop
            #add_arrow(ax, x_end[0], x_end[1], x_end[2],x_prev[0], x_prev[1], x_prev[2])
            # a = Arrow3D([x_prev[0], x_end[0]], [x_prev[1], x_end[1]], 
            #         [x_prev[2], x_end[2]], mutation_scale=20,
            #         connectionstyle="arc3,rad=-0.6",
            #         lw=3, arrowstyle="-|>", color="black",
            #             #shrinkA = 10, shrinkB = 10,
            #        zorder = 7)
            # ax.add_artist(a)

def plot_data_2d(xs, ys,  ax, lw = 2, a = 0.2, b = 0.07):
    flag = 1
    loop = False
    for j in range(len(xs)-1):
        x1, x2 = xs[j], xs[j+1]
        y1, y2 = ys[j], ys[j+1]

        ar = FancyArrowPatch(
                (x1, y1),   # start point
                (x2, y2),   # end point
                mutation_scale=20,      # controls size of arrowhead
                arrowstyle='-|>',       # style of the arrow (similar to 3D arrow)
                connectionstyle="arc3,rad=0.0",  # arc for the line connecting the points
                color='black',  # color of the arrow
                lw=lw, # line width
                zorder = 7
            )

# Add the arrow to the plot
        ax.add_patch(ar)
        if x1 == x2 and y1 == y2:
                        # Parameters for the ellipse
            if loop:
                flag+=0.5
            a = a*flag  # semi-major axis
            b = b*flag  # semi-minor axis
            loop = True
            normal_vector = np.array([1,0,0
                                     ]) # Arbitrary normal vector for the ellipse
            point = np.array([x1, y1])  # The pre-specified point through which the ellipse should pass
            refl = x1.astype(bool)
            #print(refl)
            
            # Plot the elliptical loop and get the final points for the arrow
            x_end, x_prev = plot_elliptical_loop_2d(ax, reflect = refl,
                                                 a=a, b=b, 
                                                 lw = lw,
                                                 alpha = 1,
                                                
                                                    point=point)
        else:
            loop = False

# Rotation matrix to align the ellipse with a normal vector
def rotation_matrix(v):
    """Create a rotation matrix to align the z-axis with vector v."""
    v = v / np.linalg.norm(v)  # Normalize the vector
    z = np.array([0, 0, 1])  # Target vector (z-axis)
    
    if np.allclose(v, z):  # If already aligned with z-axis, return identity matrix
        return np.eye(3)
    
    # Compute the rotation axis (cross product of z and v)
    axis = np.cross(z, v)
    axis = axis / np.linalg.norm(axis)  # Normalize the rotation axis
    
    # Compute the angle between z and v
    angle = np.arccos(np.dot(z, v))
    
    # Use the Rodrigues' rotation formula to compute the rotation matrix
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    
    R = np.eye(3) * cos_angle + np.outer(axis, axis) * (1 - cos_angle) + K * sin_angle
    return R

# Create the parametric function for the ellipse in 3D with arbitrary normal and passing through a point
def plot_elliptical_loop(ax, a=1, b=0.5, alpha = 0.5,
                         num_points=100, lw = 3,
                         normal_vector=np.array([0, 0, 1]), point=np.array([1, 1, 1])):
    t = np.linspace(0, 2 * np.pi, num_points)  # Parametric variable
    x = a * np.cos(t)
    y = b * np.sin(t)
    z = np.zeros_like(t)  # Start with the ellipse in the xy-plane
    
    # Combine x, y, z into a 3D array
    ellipse_points = np.vstack([x, y, z]).T
    
    # Apply the rotation to align the ellipse with the given normal vector
    R = rotation_matrix(normal_vector)  # Get the rotation matrix
    rotated_points = (R @ ellipse_points.T).T  # Apply the rotation to each point
    
    # Translate the ellipse such that the start and end points are the specified point
    translation = point - rotated_points[0]  # Move the first point to the specified point
    translated_points = rotated_points + translation  # Apply the translation
    
    # Plot the translated and rotated ellipse
    ax.plot(translated_points[:, 0], translated_points[:, 1], translated_points[:, 2], 
            color='black', label='Elliptical Loop', lw = lw, alpha = alpha)
    
    # Return the final point for the arrow (the last point and the second-to-last point)
    return translated_points[-1], translated_points[-2]  # Last point and the previous point

def plot_elliptical_loop_2d(ax, a=1, b=0.5, alpha = 0.5,
                         num_points=100, lw = 3,reflect = False,
                         point=np.array([1, 1, 1])):
    t = np.linspace(0, 2 * np.pi, num_points)  # Parametric variable
    x = a * np.cos(t)
    y = b * np.sin(t)
    
    # Combine x, y, z into a 3D array
    ellipse_points = np.vstack([x, y]).T
    
    rotated_points = (ellipse_points)  # Apply the rotation to each point
    
    # Translate the ellipse such that the start and end points are the specified point
    
    if reflect:
        translation = point - rotated_points[int(num_points/2)]  # Move the first point to the specified point
    else:
        translation = point - rotated_points[0]
    translated_points = rotated_points + translation  # Apply the translation
    
    # Plot the translated and rotated ellipse
    ax.plot(translated_points[:, 0], translated_points[:, 1], 
            color='black', label='Elliptical Loop', lw = lw, alpha = alpha)
    
    # Return the final point for the arrow (the last point and the second-to-last point)
    return translated_points[-1], translated_points[-2]  # Last point and the previous point

# Function to add an arrow in 3D using FancyArrowPatch with arrowstyle="-|>"
def add_arrow(ax, x_start, y_start, z_start, x_end, y_end, z_end):
    # # Calculate direction vector
    # dx = x_end - x_start
    # dy = y_end - y_start
    # dz = z_end - z_start
    
    # # Normalize the vector to prevent it from being too long
    # length = np.sqrt(dx**2 + dy**2 + dz**2)
    # dx, dy, dz = dx / length, dy / length, dz / length
    
    # Create the arrow line (as a plot for visibility)
    #ax.plot([x_start, x_end], [y_start, y_end], [z_start, z_end], color='r')

    # Now create the arrowhead using FancyArrowPatch, need to project to 3D
    arrow = Arrow3D([x_start, x_end], [y_start, y_end],
                    [z_start, z_end],
                    mutation_scale=20,
                    #connectionstyle="arc3,rad=0.1",
                    lw=4, arrowstyle="-|>", color="black",
                   zorder = 7)
    ax.add_patch(arrow)
    
    # You may need to use the 3D transformation for correct placement if using 3D plots.
def plot_triple(ddf):
    #%matplotlib qt
    sns.set(style = 'white', font_scale = 1.)
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.stats import kde
    from matplotlib.ticker import MaxNLocator
    
    # 2. Create a 3D Plot
    fig = plt.figure(figsize=(8,8), dpi = 400, constrained_layout = True)
    ax = fig.add_subplot(111, projection='3d')
    
    cnts = ddf.value_counts()
    inds = cnts.index.to_frame().values
    xx, yy, zz = inds[:, 0], inds[:, 1], inds[:, 2]
    colors = cnts.values+20
    colors = colors/np.max(colors)
    # Scatter plot of the 3D data
    ax.scatter(xx, yy, zz, s=colors*40, alpha=0.5, c=colors, vmin = 0,
               cmap = 'Greys',edgecolor='none',
               marker='o', label='3D Data')
    
    x, y, z = ddf.iloc[:, 0], ddf.iloc[:, 1], ddf.iloc[:, 2]
    
    
    # 3. Calculate and Plot Marginal KDEs using contourf
    
    # Function to create a 2D KDE and plot it with contourf
    def plot_2d_kde(x_data, y_data, zdir, offset, ax, color, cmap ='Greys', levels=10, alpha = 0.5):
        # Create 2D KDE
        k = kde.gaussian_kde([x_data, y_data])
        xi, yi = np.mgrid[(x_data.min()):(x_data.max()):x_data.size**0.5*1j,
                           (y_data.min()):(y_data.max()):y_data.size**0.5*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
    
        # Plot contourf with appropriate zdir and offset
        if zdir == 'z':
            ax.contourf(xi, yi, zi, zdir=zdir, offset=offset, cmap = cmap, levels=levels, alpha=alpha)#, colors=color)
        elif zdir == 'y':
            ax.contourf(xi, zi, yi, zdir=zdir, offset=offset, cmap = cmap, levels=levels, alpha=alpha)#, colors=color)
        elif zdir == 'x':
    
            # contour = ax.contour(zi, xi, yi, zdir=zdir, offset=offset, alpha = alpha,
            #                      levels=levels, colors='black', linewidths=0.5)
            ax.contourf(zi, xi, yi, zdir=zdir, offset=offset, cmap = cmap, levels=levels, alpha=alpha)#, colors=color)
    
    # XY Plane
    plot_2d_kde(x, y, 'z', z.min(), ax, 'r')
    
    # XZ Plane
    plot_2d_kde(x, z, 'y', y.max(), ax, 'g')
    
    # YZ Plane
    plot_2d_kde(y, z, 'x', x.min(), ax, 'purple')
    
    # Set Labels and Title
    ax.set_xlabel(cnts.index.names[0], labelpad=15, fontsize = 20)
    ax.set_ylabel(cnts.index.names[1], labelpad=15, fontsize = 20)
    ax.set_zlabel(cnts.index.names[2], labelpad=15, fontsize = 20)
    #ax.set_title('3D Volume with Marginal KDE Plots (using contourf)')
    
    # Adjust plot limits
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    ax.set_zlim([z.min(), z.max()])
    
    # Set the viewing angle
    ax.view_init(elev=20, azim=-50)
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.zaxis.set_major_locator(MaxNLocator(integer=True))
    fig.subplots_adjust(left=0.0, right=0.8, bottom=0, top=1) 
    # ax.zaxis.set_rotate_label(True)  # Disable automatic rotation
    # ax.zaxis._axinfo['label']['space_factor'] = 2 # Adjust this value as needed
    #plt.tight_layout()
    #plt.legend()
    ax.set_box_aspect(None, zoom=0.8)
    plt.show()  
    return fig

def plot_var_elim(summ,  lbl = 'Y', title = '', 
                  color = 'grey', 
                  hue = None, hue_order = None,
                  palette = None, figsize = (6,3), 
                  xlim = (-0.2, 4.2), ylim = (-0.2, 1.2),
                  lw = 10, lbl_y = 0.4, title_font = 20,
                  dpi = 300, ax = None, legend = None, **kwargs):
    if ax is None:
        plt.figure(figsize = figsize, dpi = dpi)
        ax = plt.gca()
    all_summ = pd.DataFrame(summ).set_index('proportion', append = True).index.to_frame().reset_index(drop = True).set_index('proportion', append = True).stack().reset_index()
    all_summ.columns = ['Sequence', 'prop', 'Stage', lbl]
    all_summ['Status_num'] = all_summ[lbl]
    jitters = np.linspace(-0.1,0.1,summ.shape[0])
    for i in range(summ.shape[0]):
        jitter = jitters[i]
        all_summ['Status_num'][all_summ['Sequence'] == i] +=jitter
    all_summ['prop'] = all_summ['prop']*lw
    for i in np.unique(all_summ['Sequence']):
        rel = all_summ.query('Sequence == '+str(i))
        sns.pointplot(rel, 
                      x = 'Stage', y = 'Status_num', #units = 'Sequence',
                      #color = 'grey', 
                      lw = rel['prop'].iloc[0], 
                      dodge = False, color = color, hue = hue, legend = legend,
                      hue_order = hue_order,
                     palette = palette, ax = ax, **kwargs)
    ax.set_xticks(ticks = ax.get_xticks(), 
               labels = ['0', '1', '2', '3', '4'],
              fontsize = 15)
    ax.set_yticks(ticks = [0,1.], 
               labels = [0,1], fontsize = 20)
    
    ax.tick_params(axis='both', which='both', length=0)
    ax.axvline(0.5, linestyle = '--', c = 'black', lw = 4)
    ax.axvline(2.5, linestyle = '--', c = 'black', lw = 4)
    ax.set_ylabel('$'+lbl+'$', rotation = 0, fontsize = 25)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.yaxis.set_label_coords(-0.05,lbl_y)
    ax.set_title(title, y = 1.05, fontsize = title_font)
    #plt.suptitle('$P(Y_{t=0\ldots 4}|S^+_{16})$', y = 1.05)   

