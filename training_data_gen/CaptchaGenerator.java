import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import nl.captcha.Captcha;
import nl.captcha.audio.noise.RandomNoiseProducer;
import nl.captcha.backgrounds.BackgroundProducer;
import nl.captcha.backgrounds.FlatColorBackgroundProducer;
import nl.captcha.backgrounds.GradiatedBackgroundProducer;
import nl.captcha.backgrounds.SquigglesBackgroundProducer;
import nl.captcha.backgrounds.TransparentBackgroundProducer;
import nl.captcha.gimpy.*;
import nl.captcha.noise.CurvedLineNoiseProducer;
import nl.captcha.noise.NoiseProducer;
import nl.captcha.text.producer.DefaultTextProducer;
import nl.captcha.text.producer.FiveLetterFirstNameTextProducer;

public  class CaptchaGenerator extends java.lang.Object implements java.io.Serializable
{
	
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;


	public static void generate_captcha(int image_number) throws IOException{
		//System.out.println("Starting ");
		Captcha captcha = new Captcha.Builder(200, 50)
	    .addText()
	    .addNoise()
	    .gimp()
	    .build();
		
		BufferedImage bufferedImage = captcha.getImage();
	        File outputfile = new File("/home/geetika/captcha/simpledataset/" + image_number + "_" +  captcha.getAnswer() + ".jpg");
		ImageIO.write(bufferedImage, "jpg", outputfile);
		
	}
	
	
	public static void main(String args[]) throws IOException{
		int image_number = 0;
		for(image_number = 1; image_number  < 2000000; image_number++){
			generate_captcha(1);
		
		}
		System.out.println("Done");
	}
	
}

