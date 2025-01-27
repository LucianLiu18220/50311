/**
 *Submitted for verification at Etherscan.io on 2018-08-17
*/

pragma solidity ^0.4.21;

// Generated by TokenGen and the Fabric Token platform.
// https://tokengen.io
// https://fabrictoken.io

// File: contracts/library/SafeMath.sol

/**
 * @title Safe Math
 *
 * @dev Library for safe mathematical operations.
 */
library SafeMath {
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a * b;
        assert(a == 0 || c / a == b);

        return c;
    }

    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a / b;

        return c;
    }

    function minus(uint256 a, uint256 b) internal pure returns (uint256) {
        assert(b <= a);

        return a - b;
    }

    function plus(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        assert(c >= a);

        return c;
    }
}

// File: contracts/token/ERC20Token.sol

/**
 * @dev The standard ERC20 Token contract base.
 */
contract ERC20Token {
    uint256 public totalSupply;  /* shorthand for public function and a property */
    
    function balanceOf(address _owner) public view returns (uint256 balance);
    function transfer(address _to, uint256 _value) public returns (bool success);
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success);
    function approve(address _spender, uint256 _value) public returns (bool success);
    function allowance(address _owner, address _spender) public view returns (uint256 remaining);

    event Transfer(address indexed _from, address indexed _to, uint256 _value);
    event Approval(address indexed _owner, address indexed _spender, uint256 _value);
}

// File: contracts/token/StandardToken.sol

/**
 * @title Standard Token
 *
 * @dev The standard abstract implementation of the ERC20 interface.
 */
contract StandardToken is ERC20Token {
    using SafeMath for uint256;

    string public name;
    string public symbol;
    uint8 public decimals;
    
    mapping (address => uint256) balances;
    mapping (address => mapping (address => uint256)) internal allowed;
    
    /**
     * @dev The constructor assigns the token name, symbols and decimals.
     */
    constructor(string _name, string _symbol, uint8 _decimals) internal {
        name = _name;
        symbol = _symbol;
        decimals = _decimals;
    }

    /**
     * @dev Get the balance of an address.
     *
     * @param _address The address which's balance will be checked.
     *
     * @return The current balance of the address.
     */
    function balanceOf(address _address) public view returns (uint256 balance) {
        return balances[_address];
    }

    /**
     * @dev Checks the amount of tokens that an owner allowed to a spender.
     *
     * @param _owner The address which owns the funds allowed for spending by a third-party.
     * @param _spender The third-party address that is allowed to spend the tokens.
     *
     * @return The number of tokens available to `_spender` to be spent.
     */
    function allowance(address _owner, address _spender) public view returns (uint256 remaining) {
        return allowed[_owner][_spender];
    }

    /**
     * @dev Give permission to `_spender` to spend `_value` number of tokens on your behalf.
     * E.g. You place a buy or sell order on an exchange and in that example, the 
     * `_spender` address is the address of the contract the exchange created to add your token to their 
     * website and you are `msg.sender`.
     *
     * @param _spender The address which will spend the funds.
     * @param _value The amount of tokens to be spent.
     *
     * @return Whether the approval process was successful or not.
     */
    function approve(address _spender, uint256 _value) public returns (bool) {
        allowed[msg.sender][_spender] = _value;

        emit Approval(msg.sender, _spender, _value);

        return true;
    }

    /**
     * @dev Transfers `_value` number of tokens to the `_to` address.
     *
     * @param _to The address of the recipient.
     * @param _value The number of tokens to be transferred.
     */
    function transfer(address _to, uint256 _value) public returns (bool) {
        executeTransfer(msg.sender, _to, _value);

        return true;
    }

    /**
     * @dev Allows another contract to spend tokens on behalf of the `_from` address and send them to the `_to` address.
     *
     * @param _from The address which approved you to spend tokens on their behalf.
     * @param _to The address where you want to send tokens.
     * @param _value The number of tokens to be sent.
     *
     * @return Whether the transfer was successful or not.
     */
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool) {
        require(_value <= allowed[_from][msg.sender]);
        
        allowed[_from][msg.sender] = allowed[_from][msg.sender].minus(_value);
        executeTransfer(_from, _to, _value);

        return true;
    }

    /**
     * @dev Internal function that this reused by the transfer functions
     */
    function executeTransfer(address _from, address _to, uint256 _value) internal {
        require(_to != address(0));
        require(_value != 0 && _value <= balances[_from]);
        
        balances[_from] = balances[_from].minus(_value);
        balances[_to] = balances[_to].plus(_value);

        emit Transfer(_from, _to, _value);
    }
}

// File: contracts/token/MintableToken.sol

/**
 * @title Mintable Token
 *
 * @dev Allows the creation of new tokens.
 */
contract MintableToken is StandardToken {
    /// @dev The only address allowed to mint coins
    address public minter;

    /// @dev Indicates whether the token is still mintable.
    bool public mintingDisabled = false;

    /**
     * @dev Event fired when minting is no longer allowed.
     */
    event MintingDisabled();

    /**
     * @dev Allows a function to be executed only if minting is still allowed.
     */
    modifier canMint() {
        require(!mintingDisabled);
        _;
    }

    /**
     * @dev Allows a function to be called only by the minter
     */
    modifier onlyMinter() {
        require(msg.sender == minter);
        _;
    }

    /**
     * @dev The constructor assigns the minter which is allowed to mind and disable minting
     */
    constructor(address _minter) internal {
        minter = _minter;
    }

    /**
    * @dev Creates new `_value` number of tokens and sends them to the `_to` address.
    *
    * @param _to The address which will receive the freshly minted tokens.
    * @param _value The number of tokens that will be created.
    */
    function mint(address _to, uint256 _value) onlyMinter canMint public {
        totalSupply = totalSupply.plus(_value);
        balances[_to] = balances[_to].plus(_value);

        emit Transfer(0x0, _to, _value);
    }

    /**
    * @dev Disable the minting of new tokens. Cannot be reversed.
    *
    * @return Whether or not the process was successful.
    */
    function disableMinting() onlyMinter canMint public {
        mintingDisabled = true;
       
        emit MintingDisabled();
    }
}

// File: contracts/token/BurnableToken.sol

/**
 * @title Burnable Token
 *
 * @dev Allows tokens to be destroyed.
 */
contract BurnableToken is StandardToken {
    /**
     * @dev Event fired when tokens are burned.
     *
     * @param _from The address from which tokens will be removed.
     * @param _value The number of tokens to be destroyed.
     */
    event Burn(address indexed _from, uint256 _value);

    /**
     * @dev Burnes `_value` number of tokens.
     *
     * @param _value The number of tokens that will be burned.
     */
    function burn(uint256 _value) public {
        require(_value != 0);

        address burner = msg.sender;
        require(_value <= balances[burner]);

        balances[burner] = balances[burner].minus(_value);
        totalSupply = totalSupply.minus(_value);

        emit Burn(burner, _value);
        emit Transfer(burner, address(0), _value);
    }
}

// File: contracts/trait/HasOwner.sol

/**
 * @title HasOwner
 *
 * @dev Allows for exclusive access to certain functionality.
 */
contract HasOwner {
    // The current owner.
    address public owner;

    // Conditionally the new owner.
    address public newOwner;

    /**
     * @dev The constructor.
     *
     * @param _owner The address of the owner.
     */
    constructor(address _owner) public {
        owner = _owner;
    }

    /** 
     * @dev Access control modifier that allows only the current owner to call the function.
     */
    modifier onlyOwner {
        require(msg.sender == owner);
        _;
    }

    /**
     * @dev The event is fired when the current owner is changed.
     *
     * @param _oldOwner The address of the previous owner.
     * @param _newOwner The address of the new owner.
     */
    event OwnershipTransfer(address indexed _oldOwner, address indexed _newOwner);

    /**
     * @dev Transfering the ownership is a two-step process, as we prepare
     * for the transfer by setting `newOwner` and requiring `newOwner` to accept
     * the transfer. This prevents accidental lock-out if something goes wrong
     * when passing the `newOwner` address.
     *
     * @param _newOwner The address of the proposed new owner.
     */
    function transferOwnership(address _newOwner) public onlyOwner {
        newOwner = _newOwner;
    }
 
    /**
     * @dev The `newOwner` finishes the ownership transfer process by accepting the
     * ownership.
     */
    function acceptOwnership() public {
        require(msg.sender == newOwner);

        emit OwnershipTransfer(owner, newOwner);

        owner = newOwner;
    }
}

// File: contracts/token/PausableToken.sol

/**
 * @title Pausable Token
 *
 * @dev Allows you to pause/unpause transfers of your token.
 **/
contract PausableToken is StandardToken, HasOwner {

    /// Indicates whether the token contract is paused or not.
    bool public paused = false;

    /**
     * @dev Event fired when the token contracts gets paused.
     */
    event Pause();

    /**
     * @dev Event fired when the token contracts gets unpaused.
     */
    event Unpause();

    /**
     * @dev Allows a function to be called only when the token contract is not paused.
     */
    modifier whenNotPaused() {
        require(!paused);
        _;
    }

    /**
     * @dev Pauses the token contract.
     */
    function pause() onlyOwner whenNotPaused public {
        paused = true;
        emit Pause();
    }

    /**
     * @dev Unpauses the token contract.
     */
    function unpause() onlyOwner public {
        require(paused);

        paused = false;
        emit Unpause();
    }

    /// Overrides of the standard token's functions to add the paused/unpaused functionality.

    function transfer(address _to, uint256 _value) public whenNotPaused returns (bool) {
        return super.transfer(_to, _value);
    }

    function approve(address _spender, uint256 _value) public whenNotPaused returns (bool) {
        return super.approve(_spender, _value);
    }

    function transferFrom(address _from, address _to, uint256 _value) public whenNotPaused returns (bool) {
        return super.transferFrom(_from, _to, _value);
    }
}

// File: contracts/fundraiser/AbstractFundraiser.sol

contract AbstractFundraiser {
    /// The ERC20 token contract.
    ERC20Token public token;

    /**
     * @dev The event fires every time a new buyer enters the fundraiser.
     *
     * @param _address The address of the buyer.
     * @param _ethers The number of ethers funded.
     * @param _tokens The number of tokens purchased.
     */
    event FundsReceived(address indexed _address, uint _ethers, uint _tokens);


    /**
     * @dev The initialization method for the token
     *
     * @param _token The address of the token of the fundraiser
     */
    function initializeFundraiserToken(address _token) internal
    {
        token = ERC20Token(_token);
    }

    /**
     * @dev The default function which is executed when someone sends funds to this contract address.
     */
    function() public payable {
        receiveFunds(msg.sender, msg.value);
    }

    /**
     * @dev this overridable function returns the current conversion rate for the fundraiser
     */
    function getConversionRate() public view returns (uint256);

    /**
     * @dev checks whether the fundraiser passed `endTime`.
     *
     * @return whether the fundraiser has ended.
     */
    function hasEnded() public view returns (bool);

    /**
     * @dev Create and sends tokens to `_address` considering amount funded and `conversionRate`.
     *
     * @param _address The address of the receiver of tokens.
     * @param _amount The amount of received funds in ether.
     */
    function receiveFunds(address _address, uint256 _amount) internal;
    
    /**
     * @dev It throws an exception if the transaction does not meet the preconditions.
     */
    function validateTransaction() internal view;
    
    /**
     * @dev this overridable function makes and handles tokens to buyers
     */
    function handleTokens(address _address, uint256 _tokens) internal;

    /**
     * @dev this overridable function forwards the funds (if necessary) to a vault or directly to the beneficiary
     */
    function handleFunds(address _address, uint256 _ethers) internal;

}

// File: contracts/fundraiser/BasicFundraiser.sol

/**
 * @title Basic Fundraiser
 *
 * @dev An abstract contract that is a base for fundraisers. 
 * It implements a generic procedure for handling received funds:
 * 1. Validates the transaciton preconditions
 * 2. Calculates the amount of tokens based on the conversion rate.
 * 3. Delegate the handling of the tokens (mint, transfer or conjure)
 * 4. Delegate the handling of the funds
 * 5. Emit event for received funds
 */
contract BasicFundraiser is HasOwner, AbstractFundraiser {
    using SafeMath for uint256;

    // The number of decimals for the token.
    uint8 constant DECIMALS = 18;  // Enforced

    // Decimal factor for multiplication purposes.
    uint256 constant DECIMALS_FACTOR = 10 ** uint256(DECIMALS);

    /// The start time of the fundraiser - Unix timestamp.
    uint256 public startTime;

    /// The end time of the fundraiser - Unix timestamp.
    uint256 public endTime;

    /// The address where funds collected will be sent.
    address public beneficiary;

    /// The conversion rate with decimals difference adjustment,
    /// When converion rate is lower than 1 (inversed), the function calculateTokens() should use division
    uint256 public conversionRate;

    /// The total amount of ether raised.
    uint256 public totalRaised;

    /**
     * @dev The event fires when the number of token conversion rate has changed.
     *
     * @param _conversionRate The new number of tokens per 1 ether.
     */
    event ConversionRateChanged(uint _conversionRate);

    /**
     * @dev The basic fundraiser initialization method.
     *
     * @param _startTime The start time of the fundraiser - Unix timestamp.
     * @param _endTime The end time of the fundraiser - Unix timestamp.
     * @param _conversionRate The number of tokens create for 1 ETH funded.
     * @param _beneficiary The address which will receive the funds gathered by the fundraiser.
     */
    function initializeBasicFundraiser(
        uint256 _startTime,
        uint256 _endTime,
        uint256 _conversionRate,
        address _beneficiary
    )
        internal
    {
        require(_endTime >= _startTime);
        require(_conversionRate > 0);
        require(_beneficiary != address(0));

        startTime = _startTime;
        endTime = _endTime;
        conversionRate = _conversionRate;
        beneficiary = _beneficiary;
    }

    /**
     * @dev Sets the new conversion rate
     *
     * @param _conversionRate New conversion rate
     */
    function setConversionRate(uint256 _conversionRate) public onlyOwner {
        require(_conversionRate > 0);

        conversionRate = _conversionRate;

        emit ConversionRateChanged(_conversionRate);
    }

    /**
     * @dev Sets The beneficiary of the fundraiser.
     *
     * @param _beneficiary The address of the beneficiary.
     */
    function setBeneficiary(address _beneficiary) public onlyOwner {
        require(_beneficiary != address(0));

        beneficiary = _beneficiary;
    }

    /**
     * @dev Create and sends tokens to `_address` considering amount funded and `conversionRate`.
     *
     * @param _address The address of the receiver of tokens.
     * @param _amount The amount of received funds in ether.
     */
    function receiveFunds(address _address, uint256 _amount) internal {
        validateTransaction();

        uint256 tokens = calculateTokens(_amount);
        require(tokens > 0);

        totalRaised = totalRaised.plus(_amount);
        handleTokens(_address, tokens);
        handleFunds(_address, _amount);

        emit FundsReceived(_address, msg.value, tokens);
    }

    /**
     * @dev this overridable function returns the current conversion rate for the fundraiser
     */
    function getConversionRate() public view returns (uint256) {
        return conversionRate;
    }

    /**
     * @dev this overridable function that calculates the tokens based on the ether amount
     */
    function calculateTokens(uint256 _amount) internal view returns(uint256 tokens) {
        tokens = _amount.mul(getConversionRate());
    }

    /**
     * @dev It throws an exception if the transaction does not meet the preconditions.
     */
    function validateTransaction() internal view {
        require(msg.value != 0);
        require(now >= startTime && now < endTime);
    }

    /**
     * @dev checks whether the fundraiser passed `endtime`.
     *
     * @return whether the fundraiser is passed its deadline or not.
     */
    function hasEnded() public view returns (bool) {
        return now >= endTime;
    }
}

// File: contracts/token/StandardMintableToken.sol

contract StandardMintableToken is MintableToken {
    constructor(address _minter, string _name, string _symbol, uint8 _decimals)
        StandardToken(_name, _symbol, _decimals)
        MintableToken(_minter)
        public
    {
    }
}

// File: contracts/fundraiser/MintableTokenFundraiser.sol

/**
 * @title Fundraiser With Mintable Token
 */
contract MintableTokenFundraiser is BasicFundraiser {
    /**
     * @dev The initialization method that creates a new mintable token.
     *
     * @param _name Token name
     * @param _symbol Token symbol
     * @param _decimals Token decimals
     */
    function initializeMintableTokenFundraiser(string _name, string _symbol, uint8 _decimals) internal {
        token = new StandardMintableToken(
            address(this), // The fundraiser is the token minter
            _name,
            _symbol,
            _decimals
        );
    }

    /**
     * @dev Mint the specific amount tokens
     */
    function handleTokens(address _address, uint256 _tokens) internal {
        MintableToken(token).mint(_address, _tokens);
    }
}

// File: contracts/fundraiser/GasPriceLimitFundraiser.sol

/**
 * @title GasPriceLimitFundraiser
 *
 * @dev This fundraiser allows to set gas price limit for the participants in the fundraiser
 */
contract GasPriceLimitFundraiser is HasOwner, BasicFundraiser {
    uint256 public gasPriceLimit;

    event GasPriceLimitChanged(uint256 gasPriceLimit);

    /**
     * @dev This function puts the initial gas limit
     */
    function initializeGasPriceLimitFundraiser(uint256 _gasPriceLimit) internal {
        gasPriceLimit = _gasPriceLimit;
    }

    /**
     * @dev This function allows the owner to change the gas limit any time during the fundraiser
     */
    function changeGasPriceLimit(uint256 _gasPriceLimit) onlyOwner() public {
        gasPriceLimit = _gasPriceLimit;

        emit GasPriceLimitChanged(_gasPriceLimit);
    }

    /**
     * @dev The transaction is valid if the gas price limit is lifted-off or the transaction meets the requirement
     */
    function validateTransaction() internal view {
        require(gasPriceLimit == 0 || tx.gasprice <= gasPriceLimit);

        return super.validateTransaction();
    }
}

// File: contracts/fundraiser/ForwardFundsFundraiser.sol

/**
 * @title Forward Funds to Beneficiary Fundraiser
 *
 * @dev This contract forwards the funds received to the beneficiary.
 */
contract ForwardFundsFundraiser is BasicFundraiser {
    /**
     * @dev Forward funds directly to beneficiary
     */
    function handleFunds(address, uint256 _ethers) internal {
        // Forward the funds directly to the beneficiary
        beneficiary.transfer(_ethers);
    }
}

// File: contracts/Fundraiser.sol

/**
 * @title RealDirectToken
 */
 
contract RealDirectToken is MintableToken, BurnableToken, PausableToken {
  constructor(address _owner, address _minter)
    StandardToken(
      "Real Direct Token",   // Token name
      "RDT", // Token symbol
      18  // Token decimals
    )
    HasOwner(_owner)
    MintableToken(_minter)
    public
  {
  }
}



/**
 * @title RealDirectTokenFundraiser
 */

contract RealDirectTokenFundraiser is MintableTokenFundraiser, ForwardFundsFundraiser, GasPriceLimitFundraiser {
  

  constructor()
    HasOwner(msg.sender)
    public
  {
    token = new RealDirectToken(
      msg.sender,  // Owner
      address(this)  // The fundraiser is the minter
    );

    

    initializeBasicFundraiser(
      1534291200, // Start date = 15 Aug 2018 00:00 UTC
      1544918340,  // End date = 15 Dec 2018 23:59 UTC
      20000, // Conversion rate = 20000 RDT per 1 ether
      0xEcB3c79EB0A9f539340adE65e8823CE8d248fbad     // Beneficiary
    );

    

    initializeGasPriceLimitFundraiser(
        50000000000 // Gas price limit in wei
    );

    

    

    
    
    
  }
  
}