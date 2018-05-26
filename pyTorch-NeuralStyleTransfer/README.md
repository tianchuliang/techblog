# Neural Style Transfer with PyTorch

## Basic ingredients: 
1. Convolutional Neural Networks (VGG)
2. Feature space provided by convolutional and pooling layers 

## Math
(reference: A Neural Algorithm of Artistic Style)
### Components:
To put it simply, we want to use the same CNN and extract the content information from image p and the style information from image q and somehow combine them together. 

More specifically, to get an accurate representation of the image p content, we formulate a loss of content, L_content(p,x), where p is the original input content image, and x is some white noise input image. By minimizing the L_content(p,x), we are essentially reforming x, so that it goes from white noise to something that looks like p, in terms of content. 

To get an accurate representation of the image q style, we formulate a loss of style, L_style(q,x), where q is the original input style image, and x again is some white noise input image. By minimizing the L_style(q,x), we are reforming x to have similar "styles" of image q. 

So, how do we represent content loss? If you really think about it, as the content image p passes through convolutional layers and pooling layers, it's content or pixels are being filtered through filters and max poolers, and eventually end up in some feature spaces. If we can morph the white noise image x to a similar position close to p in the feature space, then the white noise image x must have similar content or pixels. 

So, let's say for a given convolutional layer N_l, it has m filters, then we can flatten each filter's 2D features as 1D vectors of length n (filter height * filter width); so, for convolutional layer N_l, the content of image p can be represented as its feature space values F(l,p). Similarly, when we pass white noise image x through N_l, its content has feature values of F(l,x). Note, F(l, . ) is a matrix of dimension m \* n. 

Now that we can repesent content of p and x in terms of their feature values, we can define the following: 

L_content(p,x,l) = 1/2 \* MSE(F(l,x) - F(l,p))  

Style is a little bit different. From an intuitive perspective, styles are not about what's in the painting, but how are things represented, i.e edges, colors, forms, and composition. These elements are more related to each convolutional filter's feature correlation with each other. Remember, in CNNs, the filters are extracting features like lines, depth, shades, etc. So for example, in Picasso's Cubsim's paintings, how are angular edges relate to each other on the painting, how are simplistic colors relate to each other on the painting, etc. Key word, correlation. Recall when we pass an image through convlutional layer N_l, the feature values F(. , .) are of dimension m \* n, where m is the number of filters in N_l and n is the flattened dimension of each filter, then to find out how each feature relate to each other, we compute the dot product of F( . , .) with itself, and we have a Gram matrix G, which has the dimension of m * m. 

So, L_style(q,x) = 1/(2mn)^2 * MSE(G(l,q) - G(l,x))

Now, how does one combine the style and content? 

We simply start with a white noise painting x, and pass it through the CNN. When x is going through a layer from which we want to capture the content of p, we calculate the L_content; when x is going through a layer from which we want to capture the style of q, we calculate the L_style. In each iteration, we assemble the total loss 

L = a * L_content + b * L_style, where a and b are weighting factors. 

And we change values of x by subtracting the partial derivative (gradients) of error on x's pixels. 

With a few hundred iterations, x should end up like something pretty cool. 

